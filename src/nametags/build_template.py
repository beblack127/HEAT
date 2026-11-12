"""app_v4.html(디자인 원본)에 Leaflet을 넣어 단독 지도 템플릿을 만든다. 사용: python3 build_template.py 출력.html"""
import sys, pathlib
here = pathlib.Path(__file__).parent
t = (here / "app_v4.html").read_text(encoding="utf-8")
t = t.replace("__LEAFLET_CSS__", (here / "leaflet_css.html").read_text(encoding="utf-8"))
t = t.replace("__LEAFLET_JS__", (here / "leaflet_js.html").read_text(encoding="utf-8"))
pathlib.Path(sys.argv[1]).write_text(t, encoding="utf-8")
