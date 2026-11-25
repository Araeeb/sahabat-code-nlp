# BENCHMARKS — أرقام من تشغيلنا

هذا الملف يتولد تلقائياً من `python evaluate.py`. لا تعدّل الأرقام يدوياً.

## 0. خط الأساس مقابل الموديل المضبوط

- TF-IDF + LinearSVC: **0.486**
- CAMeLBERT: **0.579**
- الفرق: **9.3** نقطة

## 1. التصنيف على مجموعة الاختبار المجمدة

- عدد الأوصاف: 12
- Macro-F1: **0.579** (فترة ثقة 95%: 0.253 – 0.832)

### التقييم حسب الشرائح

| slice    | value   |   n |   accuracy |   ci_low |   ci_high |
|:---------|:--------|----:|-----------:|---------:|----------:|
| language | ar      |   9 |       0.78 |     0.44 |      1    |
| language | en      |   3 |       0    |     0    |      0    |
| dialect  | en      |   3 |       0    |     0    |      0    |
| dialect  | msa     |   8 |       0.88 |     0.62 |      1    |
| dialect  | saudi   |   1 |       0    |     0    |      0    |
| length   | long    |   4 |       0.25 |     0    |      0.75 |
| length   | short   |   8 |       0.75 |     0.38 |      1    |

### الأخطاء (للقراءة اليدوية وتصنيف السبب)

| text                                                                 | true     | pred     | reason   |
|:---------------------------------------------------------------------|:---------|:---------|:---------|
| نظام يوريك المواقف الفاضية والمشغولة على طول                         | app      | ai       |          |
| Mobile app for tracking student attendance using QR codes            | app      | ai       |          |
| Flutter app where students scan a QR code to check in to lectures    | app      | security |          |
| نظام ينبه الطلاب من الروابط الاحتيالية قبل فتحها                     | security | ai       |          |
| Password strength checker that warns about weak and reused passwords | security | ai       |          |

## 2. البحث بالمعنى

عدد الأسئلة: 8

| stage                 | queries   |   recall@10 |   MRR@10 |
|:----------------------|:----------|------------:|---------:|
| FAISS only            | ar        |           1 |     0.57 |
| FAISS only            | en        |           1 |     1    |
| FAISS only            | all       |           1 |     0.73 |
| FAISS + cross-encoder | ar        |           1 |     0.65 |
| FAISS + cross-encoder | en        |           1 |     0.83 |
| FAISS + cross-encoder | all       |           1 |     0.72 |

الفرق بين صفوف `ar` و `en` هو فجوة اللغة (cross-lingual gap).

## 3. السرعة على CPU (سلّم التحسين)

| rung                 |   p50 ms |   p99 ms |   macro-F1 |   size MB |
|:---------------------|---------:|---------:|-----------:|----------:|
| fp32 @512 padded     |   1547   |   2499.9 |      0.579 |     436.4 |
| fp32 dynamic padding |    117.5 |    244.5 |      0.579 |     436.4 |
| INT8 dynamic         |     43   |    121.6 |      0.226 |     179.9 |
