# artifacts/

يتعبأ هذا المجلد بعد التشغيل:
- `python train.py` يحفظ الموديل المضبوط، وخط الأساس، وفهرس البحث مع الـ manifest، وتوقعات مجموعة الاختبار.
- `python evaluate.py` يولّد `BENCHMARKS.md` و `evaluation.json` من أرقام تشغيلك.

ملفات الموديلات الكبيرة ما ترفع على GitHub (مستثناة في `.gitignore`)، لكن **ارفعي** `BENCHMARKS.md` و `evaluation.json` و `train_metrics.json` و `test_predictions.csv` لأنها دليل الأرقام.
