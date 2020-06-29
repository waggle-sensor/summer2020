#!/usr/bin/env python3
# Analyze KAIST scene text dataset, from http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database

import glob
import lxml.etree as ET

DATA = "./data/"
OUTPUT = "./output/"


def main():
    annots = glob.glob(DATA + "**/*.xml", recursive=True)

    char_freq = dict()
    for xml in annots:
        root = ET.parse(xml).getroot()

        try:
            words = root.find("image").find("words").findall(".//word")
        except AttributeError:
            continue

        for word in words:
            chars = word.findall(".//character")
            for char in chars:
                char_val = char.attrib["char"]
                if char_val in char_freq.keys():
                    char_freq[char_val] += 1
                else:
                    char_freq[char_val] = 0

    sorted_freq = {
        k: v
        for k, v in sorted(char_freq.items(), key=lambda item: item[1], reverse=True)
    }

    with open(OUTPUT + "freq.txt", "w+") as out:
        for k, v in sorted_freq.items():
            out.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
