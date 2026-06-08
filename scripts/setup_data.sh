#!/usr/bin/env bash
set -e
mkdir -p data
git clone --depth 1 https://github.com/Franck-Dernoncourt/pubmed-rct.git /tmp/pubmed-rct
cp -r /tmp/pubmed-rct/PubMed_20k_RCT data/
rm -rf /tmp/pubmed-rct
echo "Data ready in data/PubMed_20k_RCT/"
