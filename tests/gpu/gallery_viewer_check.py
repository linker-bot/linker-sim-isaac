"""The ycb-grasp scene in its window, buttons pressed by a script: drop an object, grasp it,
reset. Saves screenshots into the directory given and prints one JSON line."""

import json
import sys
from pathlib import Path

from PIL import Image

from linker_lab import gallery

play = gallery.load("ycb-grasp")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
report = {"states": [], "speeds": []}
OBJECT = "005_tomato_soup_can"


def script(scene, viz):
    decision = script.decisions = getattr(script, "decisions", -1) + 1
    if not report["states"] or report["states"][-1] != scene.state:
        report["states"].append(scene.state)
        frame = viz._viewer.get_frame(render_ui=True).numpy()
        Image.fromarray(frame).save(out / f"{len(report['states']):02d}_{scene.state}.png")
    if decision % 60 == 59:
        report["speeds"].append(round(scene.speed, 2))
    if decision == 30:
        scene.press("drop", next(i for i, o in enumerate(scene.items) if o["name"] == OBJECT))
    if scene.state == "ready":
        scene.press("grasp")
    if scene.state == "done" and "result" not in report:
        report["result"] = scene.message
        script.done_at = decision
    if "result" in report and decision == script.done_at + 60:
        scene.press("reset")
    if "result" in report and decision == script.done_at + 62:
        return False
    return decision < 2000


play.run(gallery.fetch("ycb-grasp"), script)
print("REPORT " + json.dumps(report), flush=True)
