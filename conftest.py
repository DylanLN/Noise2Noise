"""pytest 根配置：把项目根目录加入 sys.path，使 tests 能 import 根模块。"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
