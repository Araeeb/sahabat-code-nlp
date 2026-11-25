# ☁️ سحابة كود — Sahabat Code

**An Arabic/English NLP platform for Computer Science student projects**
**منصة لمشاريع طلاب كلية الحاسب، مدعومة بمعالجة اللغة الطبيعية بالعربي والإنجليزي**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Araeeb/sahabat-code-nlp/blob/main/notebooks/Run_in_Colab.ipynb)

> المشروع النهائي لمقرر **SDA-AIE-211: Natural Language Processing with Transformers**.
> كل لاب من 1 إلى 7 صار **مكوّن** في المنصة، والمشروع هو دمجها في خدمة وحدة.

---

## 💡 الفكرة

مشاريع الطلاب متفرقة وتضيع بعد التخرج. **سحابة كود** تجمعها، والطالب يرفع وصف مشروعه بأي لغة أو لهجة، والمنصة:

| | الميزة | المكوّن | من لاب |
|---|---|---|---|
| 🧹 | تنظّف النص وتخفي الإيميلات وأرقام الجوال | `sahabat/preprocessing.py` | 1، 4 |
| 🏷️ | تحدد المجال: ذكاء اصطناعي، تطبيقات، أمن سيبراني | `sahabat/classifier.py` | 3A |
| 🔍 | تطلع التقنيات المذكورة | `sahabat/entities.py` | 3B |
| 🧭 | تقترح مشاريع مشابهة بالمعنى | `sahabat/search.py` | 5 |
| 📊 | تقيس نفسها بصدق: شرائح، فترات ثقة، أخطاء، سرعة | `sahabat/evaluation.py`، `evaluate.py` | 6، 7 |

**الفئة المستهدفة:** طلاب كلية الحاسب، والطلاب الجدد اللي يدورون أفكار، والدكاترة، والشركات اللي تدور على مواهب.

---

## 🏗️ البنية

```
وصف المشروع
   │
   ▼
preprocess()  ── إخفاء PII ← توحيد العربي ← المسافات   (نسخة واحدة + canary عند التشغيل)
   │
   ├──► Classifier (CAMeLBERT مضبوط)      → المجال + الثقة
   ├──► extract_technologies()            → التقنيات
   └──► ProjectSearch (FAISS + manifest)  → مشاريع مشابهة
   │
   ▼
SahabatPipeline.analyze()  →  app.py (واجهة العرض)  /  api.py (HTTP)
```

```
sahabat-code-nlp/
├── sahabat/                ← كود المنصة
│   ├── config.py           ← كل الإعدادات والموديلات
│   ├── preprocessing.py    ← مصدر المعالجة الوحيد + canary
│   ├── data.py             ← تحميل البيانات + تقسيم بدون تسرّب
│   ├── classifier.py       ← خط الأساس + CAMeLBERT
│   ├── entities.py         ← استخراج التقنيات + محاذاة التصنيفات
│   ├── search.py           ← البحث بالمعنى + manifest
│   ├── evaluation.py       ← المقاييس والشرائح وفترات الثقة
│   └── pipeline.py         ← يدمج كل المكونات
├── data/
│   ├── projects.csv        ← 45 وصف مشروع
│   └── search_queries.csv  ← أسئلة مصنفة لتقييم البحث
├── train.py                ← يدرّب ويحفظ كل شي في artifacts/
├── evaluate.py             ← يولّد BENCHMARKS.md من تشغيلك
├── app.py                  ← واجهة العرض (Gradio)
├── api.py                  ← HTTP API (FastAPI)
├── tests/test_core.py      ← اختبارات المعالجة والتسرّب والمحاذاة والمقاييس
├── notebooks/
│   ├── Run_in_Colab.ipynb  ← تشغيل المشروع كامل
│   └── Labs_1_to_7.ipynb   ← اللابات خطوة خطوة (دليل التعلّم)
├── docs/                   ← DECISIONS، EVALUATION_REPORT، DEMO، مراجعة المشروع
├── artifacts/              ← مخرجات التشغيل
├── Makefile
└── requirements.txt
```

---

## 🚀 التشغيل

### في Google Colab (الأسهل)
افتحي `notebooks/Run_in_Colab.ipynb` (زر **Open in Colab** فوق)، وفعّلي GPU، وشغلي الخلايا بالترتيب.

### على جهازك
```bash
make install     # المكتبات
make test        # الاختبارات
make train       # يدرّب خط الأساس و CAMeLBERT ويبني فهرس البحث
make evaluate    # يولّد artifacts/BENCHMARKS.md
make demo        # واجهة العرض
make serve       # HTTP API على المنفذ 8000
```

بدون GPU: `make train-fast` يدرّب خط الأساس والفهرس بس.

### واجهة العرض
`python app.py` تطلع رابط `gradio.live`: الطالب يكتب وصف مشروعه، والمنصة ترجع المجال والثقة والتقنيات والبيانات المخفية والمشاريع المشابهة. خطة العرض في `docs/DEMO.md`.

### مثال API
```bash
curl -X POST localhost:8000/v1/analyze -H "Content-Type: application/json" \
     -d '{"text": "ابي اسوي تطبيق بـ Flutter يساعد الطلاب يلقون مواقف"}'
```

---

## ✅ ضمانات الجودة

- **مصدر معالجة واحد:** التدريب والمنصة والـ API كلها تستدعي `preprocess()`. ونسخة المعالجة تنحفظ مع الموديل والفهرس، والمنصة **ترفض تشتغل** لو اختلفت.
- **Canary عند التشغيل:** جملة ثابتة لازم تطلع بنفس الناتج، وإلا يتوقف التشغيل.
- **بدون تسرّب:** التقسيم حسب الفريق، مع `assert` واختبار آلي.
- **مجموعة اختبار مجمدة:** تُنشأ مرة وحدة وما تُستخدم في الضبط.
- **خط أساس صادق:** TF-IDF + LinearSVC على نفس التقسيم.
- **أرقام من تشغيلك:** `BENCHMARKS.md` يتولد آلياً، وما ينكتب يدوي.

---

## 🧾 مطابقة متطلبات المشروع النهائي

| المتطلب | وين | الحالة |
|---|---|---|
| R1 المعالجة: نسخة واحدة، إخفاء PII، canary | `preprocessing.py`، `tests/` | ✅ |
| R2 الموديلات: تصنيف مضبوط مقابل خط الأساس | `classifier.py`، `train.py` | ✅ تصنيف · ⚠️ NER: محاذاة + قاموس بدون نموذج مدرب · ❌ QA |
| R3 البحث: فهرس بـ manifest + إعادة ترتيب + Recall/MRR + فجوة اللغة | `search.py`، `evaluate.py` | ✅ |
| R4 التقييم: شرائح + فترات ثقة + أخطاء | `evaluation.py`، `EVALUATION_REPORT.md` | ✅ (قراءة الأخطاء يدوية بعد التشغيل) |
| R5 السرعة: سلّم التحسين + HTTP | `evaluate.py`، `api.py` | ✅ سلّم fp32/INT8 · ❌ p99 = 244.5 ms (الهدف 40) · ⚠️ بدون ONNX واختبار حمل HTTP |
| R6 القرارات بأدلتها | `docs/DECISIONS.md` | ✅ |
| R7 سكربتات تُعاد، اختبار مجمد، أرقام من تشغيلنا | `train.py`، `evaluate.py`، `Makefile` | ✅ |

---

## 📊 النتائج

من تشغيل `train.py` و `evaluate.py` (التفاصيل والتفسير في `docs/EVALUATION_REPORT.md`):

| البند | النتيجة |
|---|---|
| Macro-F1: خط الأساس TF-IDF | **0.486** |
| Macro-F1: CAMeLBERT | **0.579** (+9.3 نقطة، فترة ثقة 95%: 0.253 – 0.832) |
| البحث: Recall@10 / MRR@10 | **1.00 / 0.73** (عربي 0.57، إنجليزي 1.00) |
| إعادة الترتيب | MRR 0.72، ما حسّنت فما اعتمدناها |
| السرعة: fp32 تعبئة ديناميكية | p99 **244.5 ms** (أسرع 10 مرات من التعبئة لـ 512، بنفس الدقة) |
| INT8 | انهارت الدقة لـ 0.226، فرفضناه |
| التسرّب | صفر · الاختبارات 11 / 11 |

**تجربة المنصة على أوصاف جديدة:**

| الوصف | المجال | الثقة | التقنيات | أقرب مشروع |
|---|---|---|---|---|
| ابي اسوي تطبيق بـ Flutter يساعد الطلاب يلقون مواقف، كلموني 0551234567 | تطبيقات ✓ | 0.85 | Flutter (والرقم أُخفي) | بوت للتسجيل ⚠️ |
| A machine learning model that sorts recyclable waste using OpenCV | ذكاء اصطناعي ✓ | 0.87 | OpenCV، Machine Learning | سلة المهملات الذكية (عربي) ✓ |
| أداة تنبه الطلاب من رسائل التصيد في البريد الجامعي | أمن سيبراني ✓ | 0.74 | — | أداة تكشف رسائل التصيد (0.90) ✓ |

من دفتر اللابات (`notebooks/Labs_1_to_7.ipynb`):

| البند | النتيجة |
|---|---|
| Fertility لجملة «سلة مهملات ذكية تفرز النفايات» مع CAMeLBERT | **1.4** |
| حجم mBERT مقابل CAMeLBERT | **177.9M** مقابل **109.1M** (الفرق كله في جدول الكلمات) |
| اختبارات محاذاة الكيانات | **8 / 8** |
| البحث «مشروع يساعد في إعادة التدوير» | المركز الأول، **MRR = 1.0** (وإعادة الترتيب نزّلته لـ 0.5) |

---

## ⚠️ قيود صريحة
- البيانات 45 وصف لأغراض تعليمية، ومجموعة الاختبار 12 وصف، فالفترات واسعة. والإيميلات والأرقام فيها **وهمية**.
- النموذج غلط في كل الأوصاف الإنجليزية في الاختبار (3 من 3).
- استخراج الكيانات بقاموس تقنيات، مو نموذج NER مدرب.
- ما فيه نظام أسئلة وأجوبة (QA)، ولا تصدير ONNX في المنصة.

## 🔮 خطوات مستقبلية
- تدريب نموذج **NER** على بيانات مصنفة باستخدام `align_labels()` الجاهزة.
- **موجّه لهجات**: نصوص اللهجة تروح لـ CAMeLBERT-DA.
- تصنيف **متعدد** للمشاريع اللي تجمع أكثر من مجال.
- تصدير **ONNX** واختبار حمل للـ API.

---

## 🛠️ التقنيات
`Python` · `PyTorch` · `Hugging Face Transformers` · `Datasets` · `scikit-learn` · `Sentence-Transformers` · `FAISS` · `Gradio` · `FastAPI` · `pytest`

**الموديلات:** `CAMeL-Lab/bert-base-arabic-camelbert-mix` · `paraphrase-multilingual-MiniLM-L12-v2` · `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

## 👩‍💻 المطوّرة
**عرايب** — مقرر SDA-AIE-211، أكاديمية سدايا.
