"""Demo interface for Sahabat Code (the 5-minute capstone demo).

    python app.py          # web interface (Gradio), prints a public link in Colab
    python app.py --cli    # terminal version
"""
import argparse

from sahabat.pipeline import SahabatPipeline

EXAMPLES = [
    "ابي اسوي تطبيق بـ Flutter يساعد الطلاب يلقون مواقف، كلموني 0551234567",
    "A machine learning model that sorts recyclable waste using OpenCV",
    "أداة تنبه الطلاب من رسائل التصيد في البريد الجامعي",
]


def to_markdown(r):
    lines = [f"### المجال: **{r['category_ar']}** (`{r['category']}`)"]
    if r["confidence"] is not None:
        lines.append(f"الثقة: {r['confidence']:.0%}")
    lines.append(f"**التقنيات:** {', '.join(r['technologies']) or 'لا يوجد'}")
    pii = r["pii_masked"]
    if pii["emails"] or pii["phones"]:
        lines.append(f"**بيانات شخصية أُخفيت:** {pii['emails']} إيميل، {pii['phones']} جوال")
    lines.append(f"**النص بعد المعالجة:** {r['clean_text']}")
    if r["similar_projects"]:
        lines.append("\n**مشاريع مشابهة:**")
        lines += [f"- {p['text']} ({p['category']}, {p['score']})" for p in r["similar_projects"]]
    return "\n\n".join(lines)


def main(cli=False):
    pipe = SahabatPipeline()

    if cli:
        print("سحابة كود — اكتب وصف مشروعك (أو 'خروج')")
        while True:
            text = input("> ").strip()
            if text in {"خروج", "exit", "quit"}:
                break
            if text:
                print(to_markdown(pipe.analyze(text)), "\n")
        return

    import gradio as gr
    gr.Interface(
        fn=lambda text: to_markdown(pipe.analyze(text)),
        inputs=gr.Textbox(lines=3, label="وصف المشروع (عربي، إنجليزي، أو لهجة)", rtl=True),
        outputs=gr.Markdown(rtl=True),
        examples=EXAMPLES,
        title="☁️ سحابة كود",
        description="ارفع وصف مشروعك: نحدد مجاله، ونطلع التقنيات، ونقترح مشاريع مشابهة.",
    ).launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true")
    main(parser.parse_args().cli)
