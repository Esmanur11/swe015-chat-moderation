"""Run the whole pipeline: data -> baselines -> word2vec+FFN -> BERTurk -> INT8 quantization -> evaluation.

  python train_all.py           # full run (GPU recommended, ~20-30 min on a Colab T4)
  python train_all.py --quick   # 3-5 min test on a small subset: checks that every step and the app work
"""
import argparse
import os
import sys
import time


def test_app():
    """Build the Gradio app (without launching it) and send two messages through it."""
    import app
    for msg in ("Bu maçta harika oynadın, tebrikler!", "s.i.k.t.i.r git buradan"):
        _, html, log, cmp, summary, chat, logs = app.send(msg, app.DEFAULT, True, [], [])
        assert chat and logs and len(cmp) == len(app.MODELS), "app did not return a result"
        print(f"[app] {msg!r} -> {chat[-1]['label']} ({chat[-1]['conf']:.2f}) in {chat[-1]['ms']:.1f} ms; "
              f"compared {len(cmp)} models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast test run on a small subset")
    args = ap.parse_args()
    if args.quick:
        os.environ["SWE015_QUICK"] = "1"   # must be set before src.config is imported

    import train_baselines, train_w2v_ffn, train_bert, quantize, evaluate  # noqa: E401
    from src import config as C
    from src.data import load_splits

    t0 = time.time()
    load_splits()
    for step in (train_baselines, train_w2v_ffn, train_bert, quantize, evaluate):
        print(f"\n########## {step.__name__} ##########", flush=True)
        step.main()
    print("\n########## app check ##########")
    test_app()
    mins = (time.time() - t0) / 60
    if args.quick:
        print(f"\n✅ QUICK TEST PASSED in {mins:.1f} min – every step and the app work. "
              f"(Test outputs are in {C.MODELS_DIR.name}/ and {C.RESULTS_DIR.name}/, ignore their numbers.)\n"
              f"Now run the full pipeline:  python train_all.py")
    else:
        print(f"\n✅ FULL RUN FINISHED in {mins:.1f} min. Results: {C.RESULTS_DIR}/results.md\n"
              f"Next: python fill_slides.py --name ... --id ... --repo ... --video ...   and   python app.py")


if __name__ == "__main__":
    sys.exit(main())
