# REAR transformers + bitsandbytes

Репозиторий содержит материалы практической работы по воспроизведению и модификации метода REAR для задачи open-domain question answering.

В работе рассматривается статья:

**REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering**
https://arxiv.org/abs/2402.17497

Официальный репозиторий REAR:
https://github.com/RUCAIBox/REAR

## Цель работы

Цель проекта — воспроизвести ключевую логику метода REAR и проверить модификацию выбора ответа на основе post-processing/reranking-подхода.

Оригинальная реализация REAR использует `vLLM`. В данной работе inference был адаптирован под `transformers` с 8-битной загрузкой модели через `bitsandbytes`, что позволило запустить эксперимент локально на GPU с ограниченным объёмом VRAM.

## Что реализовано

В ноутбуке `REAR_transformers_bitsandbytes.ipynb` реализованы следующие режимы:

1. Direct RAG top-10;
2. Adapted REAR path-reliability;
3. Adapted REAR consistency official_first_char;
4. Adapted REAR consistency full_answer;
5. Answer-level aggregation;
6. RRF + nothing;
7. RRF + Answer-level aggregation.

## Модель и данные

Модель:

* `RUCAIBox/rear-llama-7b-hf`
* https://huggingface.co/RUCAIBox/rear-llama-7b-hf

Датасет:

* `yhao-wang/rear-eval`
* https://huggingface.co/datasets/yhao-wang/rear-eval

Модель и данные не добавлены в репозиторий из-за большого размера. Ноутбук содержит код для их загрузки и использования.

## Окружение

Эксперименты выполнялись в среде:

* Python 3.11;
* PyTorch 2.1.2 + CUDA 11.8;
* transformers 4.38.0;
* datasets 2.14.3;
* bitsandbytes 0.43.1;
* accelerate 0.27.2.

Полный список зависимостей находится в `requirements.txt`.

## Установка

Создать и активировать виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Если PyTorch с CUDA не установился автоматически, его можно поставить отдельно:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

## Запуск

Открыть ноутбук и выполнить ячейки последовательно сверху вниз.

## Основные параметры эксперимента

В работе использовались параметры, близкие к исходной реализации REAR:

* `threshold = 13`;
* `top_k documents = 10`;
* `max_new_tokens = 20`;
* `temperature = 0`;
* `top_p = 1`;
* `consistency_weight = 9`;
* `max_input_tokens = 3968`;
* `SEED = 42`.

## Основные результаты

На holdout-подвыборке из 80 примеров были получены следующие результаты:

| Метод                                        |     EM |       F1 |
| -------------------------------------------- | -----: | -------: |
| Direct RAG top-10                            | 0.1500 | 0.188429 |
| Adapted REAR path-reliability                | 0.4125 | 0.544524 |
| Adapted REAR consistency official_first_char | 0.4125 | 0.544524 |
| Adapted REAR consistency full_answer         | 0.4500 | 0.578274 |
| RRF + nothing                                | 0.4625 | 0.607857 |
| RRF + Answer-level aggregation               | 0.4625 | 0.607857 |


## Ограничения

Работа является адаптированным воспроизведением REAR. Оригинальный backend `vLLM` был заменён на `transformers`, а модель запускалась в 8-битном режиме через `bitsandbytes`. Поэтому результаты не являются полной побитовой копией оригинального эксперимента, но сохраняют основную алгоритмическую логику метода.
