# خطوات رفع المشروع على GitHub

## 1. جهّزي الملفات على جهازك
1. فكي `sahabat-code-nlp.zip`.
2. من Colab نزّلي مخرجات تشغيلك (`artifacts.zip`)، وفكيه، وحطي محتواه داخل مجلد `artifacts/` في المشروع. لازم يكون فيه:
   - `BENCHMARKS.md`
   - `evaluation.json`
   - `train_metrics.json`
   - `test_predictions.csv`
   - `test_ids.csv`
   - `search_index/manifest.json`

## 2. عدّلي قبل الرفع
- في `README.md`: غيري `YOUR_USERNAME` لاسم حسابك، و«اكتبي اسمك هنا» لاسمك.
- في `notebooks/Run_in_Colab.ipynb`: غيري `YOUR_USERNAME` في خلية `git clone`.

## 3. ارفعي
1. github.com ← **New repository** ← الاسم `sahabat-code-nlp` ← **Public** ← **Create repository**.
2. **uploading an existing file** ← اسحبي **محتوى** المجلد كامل (مو المجلد نفسه).
3. **Commit changes**.

## 4. تأكدي
- صفحة المستودع تعرض الـ README.
- زر **Open in Colab** يفتح الدفتر.
- `artifacts/BENCHMARKS.md` موجود وأرقامه تطابق التقرير.

> ملفات الموديل الكبيرة ما ترفع (مستثناة في `.gitignore`)، وتتولد من جديد بـ `python train.py`.
