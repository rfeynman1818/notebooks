#!/usr/bin/env python3
import sys, argparse
from pathlib import Path

def extract_child_prefix(geojson_name):
    name = Path(geojson_name).name
    if name.endswith(".geojson"):
        name = name[:-8]
    parts = name.split("_")
    try:
        sicd_index = parts.index("SICD")
    except ValueError:
        raise ValueError(f"'SICD' not found in name: {geojson_name}")
    if sicd_index + 2 >= len(parts):
        raise ValueError(f"Unexpected name format: {geojson_name}")
    return f"{parts[sicd_index+1]}_{parts[sicd_index+2]}", name

def find_subdir(parent_dir, child_prefix):
    candidates = [p for p in parent_dir.glob(child_prefix + "_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No subdirectory starting with {child_prefix}_ in {parent_dir}")
    candidates.sort()
    return candidates[0]

def find_xml(subdir, base_name, child_prefix):
    expected = subdir / (base_name + ".xml")
    if expected.exists():
        return expected.resolve()
    candidates = sorted(subdir.glob(f"*{child_prefix}*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No XML file matching *{child_prefix}*.xml in {subdir}")
    return candidates[0].resolve()

def process_list(list_path, parent_dir, output_path):
    parent_dir = Path(parent_dir).expanduser().resolve()
    list_path = Path(list_path).expanduser().resolve()
    output_path = Path(output_path).expanduser()

    xml_paths = []

    with list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            child_prefix, base_name = extract_child_prefix(line)
            subdir = find_subdir(parent_dir, child_prefix)
            xml_path = find_xml(subdir, base_name, child_prefix)
            xml_paths.append(str(xml_path))

    with output_path.open("w") as out:
        for p in xml_paths:
            out.write(p + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Given a .txt list of ICEYE SICD geojsons, find corresponding SICD XML files."
    )
    parser.add_argument("geojson_list", help="Path to .txt file containing geojson names (one per line)")
    parser.add_argument("parent_directory", help="Parent directory containing SLED_*_* subdirectories")
    parser.add_argument("output_txt", help="Output .txt file to write XML realpaths")
    args = parser.parse_args()

    try:
        process_list(args.geojson_list, args.parent_directory, args.output_txt)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
