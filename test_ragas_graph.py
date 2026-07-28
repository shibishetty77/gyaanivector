"""
RAGAS evaluation for the graph-based retrieval modes ("graph" and
"hybrid_graph"), mirroring test_ragas.py's structure so results are
directly comparable to the vector-only run.

Unlike test_ragas.py, this script goes through `build_retriever(...,
retrieval_mode=..., knowledge_graph=..., graph_store=...)` -- the exact
same retriever construction RAGService uses internally -- instead of
calling vector_store.similarity_search directly. That's what actually
exercises GraphStore.traverse() / extract_entities_from_query() (and, for
"hybrid_graph", RRF fusion with vector + BM25) rather than pure vector
search.

Usage:
    cd backend  # NOT required -- path setup below handles it either way
    python test_ragas_graph.py                      # runs all modes, prints comparison
    python test_ragas_graph.py --mode graph          # graph only
    python test_ragas_graph.py --mode hybrid_graph    # hybrid_graph only
    python test_ragas_graph.py --mode vector          # vector only (baseline, for comparison)
    python test_ragas_graph.py --mode graph --l3      # hallucination-focused metrics only
"""

import asyncio
import argparse
import logging
import os
import sys
import pandas as pd
from datasets import Dataset

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
if os.path.isdir(BACKEND_DIR):
    sys.path.insert(0, BACKEND_DIR)
else:
    # Script is already sitting inside backend/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import get_vector_store
from app.rag.prompt import build_prompt_template, format_docs
from app.rag.retriever import build_retriever
from app.rag.graph import get_graph_store
from app.rag.knowledge_graph import get_knowledge_graph
from app.services.llm import get_llm
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- RAGAS compatibility patch (same as test_ragas.py) --------------------
import types
try:
    import langchain_community.chat_models
    import langchain_community.llms

    if not hasattr(langchain_community.chat_models, 'vertexai'):
        langchain_community.chat_models.vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
        sys.modules["langchain_community.chat_models.vertexai"] = langchain_community.chat_models.vertexai

    if not hasattr(langchain_community.chat_models.vertexai, 'ChatVertexAI'):
        langchain_community.chat_models.vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})

    if not hasattr(langchain_community.llms, 'VertexAI'):
        langchain_community.llms.VertexAI = type("VertexAI", (object,), {})
except Exception:
    pass
# ---------------------------------------------------------------------------

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Same test set as test_ragas.py -- keep these identical so vector vs graph
# results are apples-to-apples. If you add questions here, add them to
# test_ragas.py too (or better: factor TEST_DATA out into a shared module).
TEST_DATA = [
    {
        "question": "What are the main differences between the ESP32-WROOM and ESP32-WROOM-32E modules?",
        "ground_truth": "The ESP32-WROOM-32E features the newer ESP32-D0WD-V3 chip which offers improved stability, security (like secure boot V2), and slightly better power efficiency compared to the older chip in the standard ESP32-WROOM. It also has a modified antenna design for better RF performance."
    },
    {
        "question": "What is the operating voltage range for the ESP32-WROOM-32E?",
        "ground_truth": "The operating voltage range is typically 3.0V to 3.6V."
    },
    {
        "question": "How do I connect a DHT22 temperature sensor to an ESP32?",
        "ground_truth": "Connect the VCC pin of the DHT22 to the 3.3V pin of the ESP32, GND to GND, and the Data pin to a digital GPIO pin on the ESP32. A 10k ohm pull-up resistor is typically required between VCC and Data."
    },
    {
        "question": "What is the I2C address commonly used for the LCD 1602 display?",
        "ground_truth": "The most common I2C addresses for the LCD 1602 module (when used with an I2C adapter) are 0x27 or 0x3F."
    },
    {
        "question": "What is the difference in temperature reading accuracy between DHT11 and DHT22?",
        "ground_truth": "The DHT22 is more accurate, typically offering ±0.5°C temperature accuracy, whereas the DHT11 offers ±2.0°C accuracy."
    },

    # --- Added from backend/services/master_prompt.py -- pin/wiring reference rules ---
    # Precise entity-to-entity facts (platform -> component -> pin), which is exactly
    # the structure graph retrieval should be strong at versus vector search.

    # DATA pin assignments per platform + sensor
    {
        "question": "Which pin should be used for the DATA line when connecting a DHT22 to an STM32F401?",
        "ground_truth": "The DATA pin must be PA1. PA9 and PA10 should never be used since those are the UART TX and RX pins."
    },
    {
        "question": "What DATA pin should be used for a DS18B20 sensor on an STM32F401?",
        "ground_truth": "The DATA pin must be PA0, which avoids a conflict with a DHT22 that would be on PA1."
    },
    {
        "question": "What are the I2C pins for connecting a BME280 or INA219 to an STM32F401 or STM32F303?",
        "ground_truth": "SDA connects to PB7 and SCL connects to PB6."
    },
    {
        "question": "Which pin should be avoided for DHT22 DATA on an STM32F303, and why?",
        "ground_truth": "PA6 should be avoided because it is the LED/PWM pin, and PA2/PA3 should be avoided because they are the UART TX/RX pins. The DATA pin must be PA1 instead."
    },
    {
        "question": "What DATA pin should be used for a DS18B20 sensor on an STM32F303, and what PlatformIO board is used?",
        "ground_truth": "The DATA pin must be PA0. The PlatformIO board for STM32F303 is robotdyn_blackpill_f303cc."
    },
    {
        "question": "What GPIO pin is used for DHT11 or DHT22 DATA on an ESP32?",
        "ground_truth": "GPIO pin 4 (the integer 4), defined in firmware as #define DATA_PIN 4."
    },
    {
        "question": "What pin is used for a DS18B20 sensor on an ESP32, and why that pin specifically?",
        "ground_truth": "Pin 13 (integer), defined as #define ONE_WIRE_BUS 13. It avoids conflicting with a DHT22 on pin 4 and is not a boot-sensitive pin."
    },
    {
        "question": "What are the default I2C SDA and SCL pins on an ESP32 for a BME280 or INA219?",
        "ground_truth": "SDA is GPIO21 and SCL is GPIO22."
    },
    {
        "question": "What DATA pin is used for a DHT sensor on an Arduino Uno, and how should it be written in code?",
        "ground_truth": "Pin 2, written as the plain integer 2 (#define DHT_PIN 2) — never as D2, since D2 is not a valid macro in the Arduino AVR core and causes a compile error. Pins 0 and 1 (RX/TX) must never be used for sensor data."
    },
    {
        "question": "What are the I2C SDA and SCL pins on an Arduino Uno for a BME280 or INA219?",
        "ground_truth": "SDA is A4 and SCL is A5."
    },

    # LED default pins
    {
        "question": "What is the default on-board LED pin for each supported platform (ESP32-WROOM, STM32F401, STM32F303, Arduino Uno)?",
        "ground_truth": "ESP32-WROOM uses GPIO2 (on-board blue LED). STM32F401 uses PA5 (LD2 on the Nucleo board). STM32F303 uses PA6, a PWM-capable pin (never PA5, which is the SPI SCK pin on that board). Arduino Uno uses LED_BUILTIN, which is pin 13."
    },
    {
        "question": "For an LED project, should a resistor node be added to the canvas?",
        "ground_truth": "No. For any LED project (blink, glow, PWM, fade, breathe, brightness control), a resistor node should never be added — it is omitted from both the canvas and the BOM entirely. The GPIO pin wires directly to the LED anode, and the LED cathode wires directly to GND."
    },
    {
        "question": "Does the Arduino Uno's pin 13 (LED_BUILTIN) support PWM for glow or fade effects?",
        "ground_truth": "No. Pin 13 does not support PWM on Arduino Uno. For glow, fade, breathe, or PWM brightness effects, a PWM-capable pin such as D9, D10, or D11 should be used instead."
    },

    # INA219 wiring and pull-ups
    {
        "question": "Do INA219 modules need external pull-up resistors on ESP32 or Arduino Uno?",
        "ground_truth": "No. The INA219 breakout module has built-in pull-up resistors, so no external resistors are needed on ESP32 or Arduino Uno — only 4 wires total (SDA, SCL, VCC, GND) are required."
    },
    {
        "question": "Does an STM32 need external pull-up resistors for an INA219 over I2C?",
        "ground_truth": "Yes. STM32's I2C is open-drain, so 2 external 4.7K pull-up resistors are required, for a total of 8 wires."
    },
    {
        "question": "Should the INA219's A0 and A1 address pins be wired to the MCU?",
        "ground_truth": "No. A0 and A1 are hardwired on the module PCB (default is GND, giving I2C address 0x40) and should never be wired to anything."
    },
    {
        "question": "Should the INA219's VIN+ and VIN- pins be connected to the MCU?",
        "ground_truth": "No. VIN+ and VIN- are external load measurement points only and should never be connected to the MCU."
    },
    {
        "question": "What voltage should the INA219's VCC be wired to on Arduino Uno versus ESP32 or STM32?",
        "ground_truth": "On Arduino Uno, VCC goes to 5V. On ESP32 or STM32, VCC goes to 3.3V."
    },

    # PWM pins
    {
        "question": "Which pins on the STM32F303 support PWM when using the Arduino framework's analogWrite()?",
        "ground_truth": "PA0, PA1 (TIM2, safe), PA2, PA3 (TIM2, but conflict with USART2 serial), PA6, PA7, PB0, PB1 (TIM3, safe), and PB4, PB5 (TIM3, safe)."
    },
    {
        "question": "Which pins on the STM32F401 support PWM with the Arduino framework?",
        "ground_truth": "PA0, PA1, PB0, PB1, PB6, and PB7."
    },
    {
        "question": "Which pins support PWM (analogWrite) on the Arduino Uno?",
        "ground_truth": "D3, D5, D6, D9, D10, and D11."
    },
    {
        "question": "How should PWM effects like glow or fade be implemented on STM32 or Arduino Uno using the Arduino framework?",
        "ground_truth": "Using analogWrite(pin, 0-255). HAL timer functions such as TIM_HandleTypeDef, HAL_TIM_PWM_Init, HAL_TIM_PWM_ConfigChannel, or HAL_TIM_PWM_Start should never be used, since analogWrite handles the timer/channel configuration internally."

    },

    # BME280
    {
        "question": "What must be done with the BME280's CSB pin for I2C mode to work?",
        "ground_truth": "The CSB pin must be tied to 3.3V (or VDD) — otherwise the sensor stays in SPI mode instead of I2C mode."
    },

    # Pull-up resistor values
    {
        "question": "What pull-up resistor values are recommended for DHT22 and DS18B20 data lines?",
        "ground_truth": "10K ohms for DHT22, and 4.7K ohms for DS18B20. I2C lines also typically use 4.7K ohm pull-ups."
    },
    {
        "question": "In a pull-up resistor wiring, where do the A and B pins of the resistor connect?",
        "ground_truth": "Pin A connects to the signal line (the DATA line), and pin B connects to the MCU's 3.3V (or VDD) pin — pin B should never be connected to a GPIO."
    },

    # Platform identification / aliases
    {
        "question": "What canonical platform name corresponds to mentions like 'blackpill', 'f303', or 'stm32f3'?",
        "ground_truth": "STM32F303."
    },
    {
        "question": "What canonical platform name corresponds to mentions like 'nucleo' or a generic 'stm32'?",
        "ground_truth": "STM32F401."
    },
    {
        "question": "What canonical platform name corresponds to mentions like 'esp', 'wroom', or 'espressif'?",
        "ground_truth": "ESP32-WROOM."
    },
    {
        "question": "What canonical platform name corresponds to mentions like 'atmega328' or 'uno'?",
        "ground_truth": "Arduino Uno."
    },

    # Sensor intent resolution
    {
        "question": "Which sensor is recommended when a user wants to measure voltage, current, or power consumption?",
        "ground_truth": "The INA219, used to measure voltage and current."
    },
    {
        "question": "Which sensor is recommended for reading barometric pressure, altitude, or general weather data?",
        "ground_truth": "The BME280, used to read pressure and weather data."
    },
    {
        "question": "Which sensor is recommended for monitoring humidity?",
        "ground_truth": "The DHT22, used to monitor humidity."
    },
    {
        "question": "Which sensor is the default choice for general temperature monitoring?",
        "ground_truth": "The DHT22."
    },

    # STM32F303 Alternate Function table
    {
        "question": "What is the default I2C pin pairing on the STM32F303, and what GPIO mode should I2C pins use?",
        "ground_truth": "PB6 for SCL and PB7 for SDA, using GPIO_AF4_I2C1. I2C pins must use GPIO_MODE_AF_OD (open-drain), not GPIO_MODE_AF_PP."
    },
    {
        "question": "What is the default SPI1 pin mapping on the STM32F303?",
        "ground_truth": "PA4=NSS, PA5=SCK, PA6=MISO, PA7=MOSI, using GPIO_AF5_SPI1."
    },
    {
        "question": "Which USART pin pair is recommended for debug serial on the STM32F303?",
        "ground_truth": "PA2=TX and PA3=RX, using GPIO_AF7_USART2 — this is the best choice for debug serial."
    },
    {
        "question": "On the STM32F303, which TIM2 pins are safe for PWM without conflicts, and which TIM2 pins conflict with USART2?",
        "ground_truth": "PA0 (CH1) and PA1 (CH2) are safe for PWM with no conflicts. PA2 (CH3) and PA3 (CH4) also work for PWM but conflict with USART2 TX/RX (which uses the same pins via GPIO_AF7), so using them for PWM disables the serial monitor. PA0/PA1 are preferred for TIM2 PWM."
    },
    {
        "question": "Can the STM32F303's CAN and USB peripherals be used at the same time?",
        "ground_truth": "No. USB and CAN share pins PA11 and PA12 on the STM32F303, so both cannot be used simultaneously."
    },
    {
        "question": "Which STM32F303 pins are used for SWD debugging, and can they be remapped?",
        "ground_truth": "PA13 (SWDIO) and PA14 (SWCLK). They should not be remapped unless SWD debugging is intentionally being disabled."
    },
    {
        "question": "What are the current and speed limits for pin PC13 on the STM32F303?",
        "ground_truth": "PC13 is limited to a maximum of 3 mA at 2 MHz, and should use a resistor of at least 1kΩ with GPIO_SPEED_FREQ_LOW."
    },

    # Firmware / code generation rules
    {
        "question": "Should HAL_Init() or SystemClock_Config() be called manually in STM32 firmware that uses the Arduino framework?",
        "ground_truth": "No. The Arduino core already calls HAL_Init() in its startup code before setup() runs, and it handles all clock initialization internally, so SystemClock_Config() should never be written or called."
    },
    {
        "question": "Why should an Error_Handler function never be declared in STM32 Arduino-framework firmware?",
        "ground_truth": "Because stm32_def.h already defines Error_Handler as a macro; declaring a function with the same name causes a syntax error ('expected unqualified-id before while'). An unrecoverable error should instead be signaled with while (1);."
    },
    {
        "question": "What should be used instead of HAL_TIM_SetCompare() in STM32 firmware?",
        "ground_truth": "The macro __HAL_TIM_SET_COMPARE(&htim, TIM_CHANNEL_x, value), since HAL_TIM_SetCompare() was removed from the STM32 HAL. Similarly, __HAL_TIM_GET_COMPARE should be used instead of HAL_TIM_GetCompare."
    },
    {
        "question": "Can config.h contain #include directives?",
        "ground_truth": "No. config.h must never contain any #include directive — it is a pure #define-only file with include guards. HAL includes belong only in main.cpp."
    },
    {
        "question": "How should ESP32 pin numbers be written in firmware code, versus how they appear on the wiring canvas?",
        "ground_truth": "In firmware (Arduino framework), pin numbers must be plain integers, e.g. #define DATA_PIN 4 — never identifiers like GPIO4 or GPIO_NUM_4. The 'GPIO4' label seen on the wiring canvas is only for visualization and is not a valid C identifier."
    },
    {
        "question": "Why should bare names like SSID or PASSWORD be avoided as #define names in ESP32/Arduino firmware?",
        "ground_truth": "Because SSID and PASSWORD are reserved by the WiFi library and cause compile errors; WIFI_SSID and WIFI_PASSWORD should be used instead."
    },
    {
        "question": "What is the correct #include order at the top of main.cpp?",
        "ground_truth": "#include <Arduino.h> must come first, followed by #include \"config.h\" second — this order should never be reversed."
    },
    {
        "question": "Should main.cpp ever include a main.h file?",
        "ground_truth": "No. The only project header is config.h — all declarations, typedefs, and extern variables belong there. A main.h file should never be generated or included."
    },
    {
        "question": "Should individual STM32 HAL sub-headers like stm32f3xx_hal_gpio.h be included directly in main.cpp?",
        "ground_truth": "No. Including HAL sub-headers directly causes 'HAL_StatusTypeDef does not name a type' compile errors. Including <Arduino.h> first automatically pulls in the correct master HAL header, so a second include of stm32f3xx_hal.h or stm32f4xx_hal.h should never be added."
    },
    {
        "question": "What PlatformIO platform, board, and framework should be used for each supported MCU?",
        "ground_truth": "ESP32 uses platform=espressif32, board=esp32dev, framework=arduino. STM32F401 uses platform=ststm32, board=nucleo_f401re, framework=arduino. STM32F303 uses platform=ststm32, board=robotdyn_blackpill_f303cc, framework=arduino. Arduino Uno uses platform=atmelavr, board=uno, framework=arduino."
    },
    {
        "question": "What framework should always be used for STM32 boards, and what should never be used?",
        "ground_truth": "The framework should always be 'arduino'. 'stm32cube' should never be used."
    },

    # Wire counts for reference wiring examples
    {
        "question": "How many wires are needed for a basic ESP32 to DHT22 connection, including the pull-up resistor?",
        "ground_truth": "5 wires total: Data, Power (VCC to 3.3V), Ground, plus 2 wires for the pull-up resistor (signal and VCC)."
    },
    {
        "question": "How many wires are needed to connect a BME280 to an ESP32 or STM32 over I2C, including pull-ups and the CSB pin?",
        "ground_truth": "9 wires total: SDA, SCL, VCC, GND, CSB tied to 3.3V/VDD for I2C mode, plus 2 wires each for the SDA and SCL pull-up resistors."
    },
    {
        "question": "How many wires are needed to connect an INA219 to an ESP32 or Arduino Uno?",
        "ground_truth": "4 wires only (SDA, SCL, VCC, GND) — no external pull-up resistors are needed since the module has built-in pull-ups."
    },
    {
        "question": "How many wires are needed to connect an INA219 to an STM32?",
        "ground_truth": "8 wires: SDA, SCL, VCC, GND, plus 2 wires each for two external 4.7K pull-up resistors required for STM32's open-drain I2C."
    },
]


async def build_evaluation_dataset(
    embeddings, vector_store, llm, knowledge_graph, graph_store, retrieval_mode, k=4, rrf_k=60
):
    """Same shape as test_ragas.py's build_evaluation_dataset, but routes
    retrieval through build_retriever with the given retrieval_mode -- so
    "graph" exercises GraphStore.traverse(), "hybrid_graph" exercises RRF
    fusion of vector + BM25 + graph, etc. (same modes app.config.Settings
    documents: vector | bm25 | hybrid | graph | hybrid_graph)."""
    print(f"\nGenerating answers using retrieval_mode='{retrieval_mode}'...")
    dataset_rows = []

    prompt = build_prompt_template()
    chain = prompt | llm | StrOutputParser()

    retriever = build_retriever(
        embeddings, vector_store, k, category=None,
        knowledge_graph=knowledge_graph,
        retrieval_mode=retrieval_mode,
        rrf_k=rrf_k,
        graph_store=graph_store,
    )

    for item in TEST_DATA:
        question = item["question"]

        docs = await retriever.ainvoke(question)
        context = await RunnableLambda(format_docs).ainvoke(docs)
        answer = await chain.ainvoke({"question": question, "context": context, "chat_history": []})

        contexts_list = [d.page_content for d in docs]
        retrieved_via = [d.metadata.get("retrieved_via", "?") for d in docs]

        print(f"  Q: {question[:60]}...")
        print(f"     retrieved {len(docs)} docs via: {retrieved_via}")

        dataset_rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts_list,
            "reference": item["ground_truth"],
        })

    df = pd.DataFrame(dataset_rows)
    hf_dataset = Dataset.from_pandas(df)
    return EvaluationDataset.from_hf_dataset(hf_dataset)


def run_ragas_evaluation(dataset, metrics_to_run, evaluator_llm, evaluator_embeddings, label):
    print("\n" + "=" * 50)
    print(f"MODE: {label}")
    print("=" * 50)

    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    summary = df.mean(numeric_only=True)

    print("-" * 40)
    for metric_name, score in summary.items():
        print(f"{metric_name.replace('_', ' ').title():<20} : {score:.2f}")
    print("-" * 40)
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for graph-based retrieval modes.")
    parser.add_argument(
        "--mode", choices=["vector", "graph", "hybrid_graph", "all"], default="all",
        help="Which retrieval mode to evaluate (default: all, runs and compares every mode)"
    )
    parser.add_argument("--l3", action="store_true", help="Hallucination-focused metrics only (Faithfulness + AnswerRelevancy)")
    parser.add_argument("--k", type=int, default=4, help="Number of documents/entities to retrieve (default: 4)")
    args = parser.parse_args()

    modes = ["vector", "graph", "hybrid_graph"] if args.mode == "all" else [args.mode]

    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR if os.path.isdir(BACKEND_DIR) else os.path.dirname(os.path.abspath(__file__)))
    try:
        settings = get_settings()
        embeddings = get_embeddings()
        vector_store = get_vector_store()
        llm = get_llm()
        knowledge_graph = get_knowledge_graph()
        graph_store = get_graph_store()
    finally:
        os.chdir(original_cwd)

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [Faithfulness(), AnswerRelevancy()] if args.l3 else \
        [ContextPrecision(), ContextRecall(), AnswerRelevancy(), Faithfulness()]

    all_results: dict[str, pd.Series] = {}

    for mode in modes:
        dataset = await build_evaluation_dataset(
            embeddings, vector_store, llm, knowledge_graph, graph_store,
            retrieval_mode=mode, k=args.k, rrf_k=settings.rrf_k,
        )
        summary = run_ragas_evaluation(dataset, metrics, evaluator_llm, evaluator_embeddings, mode.upper())
        all_results[mode] = summary

    if len(all_results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON ACROSS MODES")
        print("=" * 50)
        comparison_df = pd.DataFrame(all_results)
        print(comparison_df.round(3).to_string())


if __name__ == "__main__":
    asyncio.run(main())