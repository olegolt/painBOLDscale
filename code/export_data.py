"""Create the shareable ``data/`` folder from the private study files (run once, by the data owners).

    python code/export_data.py --input /path/to/study --results /path/to/study/Results --out data \
        --id-map /path/to/private/id_map.tsv

The ID map links pseudonyms to original IDs. Keep it out of the repository. Afterwards run
``python code/check_release.py data code``.
"""

import argparse

from painpred.config import Project
from painpred.export import export_public_data

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", required=True, help="private study folder")
parser.add_argument("--results", help="results folder of the full pipeline run (default: <input>/Results)")
parser.add_argument("--out", default="data", help="shared data folder to create")
parser.add_argument("--id-map", required=True, help="where to write the private pseudonym key (outside the repository)")
parser.add_argument("--third-party", action="store_true", help="also copy atlas and NPS weights (only if their licences allow it)")
args = parser.parse_args()
export_public_data(Project(args.input, args.results), args.out, args.id_map, include_third_party=args.third_party)
