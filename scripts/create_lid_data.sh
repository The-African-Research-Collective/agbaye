#!/bin/bash

FLORES_AFRICAN_LANGUAGES=(
    "afr_Latn" "amh_Ethi" "bam_Latn" "bem_Latn" "cjk_Latn"
    "dik_Latn" "ewe_Latn" "fon_Latn" "fuv_Latn" "gaz_Latn"
    "hau_Latn" "ibo_Latn" "kab_Latn" "kik_Latn" "kin_Latn"
    "kmb_Latn" "knc_Latn" "kon_Latn" "lua_Latn" "luo_Latn"
    "lug_Latn" "mos_Latn" "nso_Latn" "nya_Latn" "plt_Latn"
    "run_Latn" "sna_Latn" "som_Latn" "sot_Latn" "ssw_Latn"
    "swa_Latn" "taq_Latn" "taq_Tfng" "tir_Ethi" "tsn_Latn"
    "tso_Latn" "twi_Latn" "umb_Latn" "wol_Latn" "xho_Latn"
    "yor_Latn" "zul_Latn"
)

function filter_african_languages() {
    input="$1"
    output="$2"

    AFRICAN_LANGUAGES_FILTER=$(IFS="|"; echo "${FLORES_AFRICAN_LANGUAGES[*]}")
    awk -v codes="$AFRICAN_LANGUAGES_FILTER" -F'\t' '$2 ~ codes' $1 > $2
}

function create_lid_training_data() {
    data_dir="$1"
    macrolanguages="$2"

    mkdir -p $data_dir

    if [ -f "$data_dir/lid201-data-unsampled.tsv" ]; then
        :
    elif [ ! -f "$data_dir/lid201-data-unsampled.tsv.gz" ]; then
        wget https://data.statmt.org/lid/lid201-data-unsampled.tsv.gz -P $data_dir
        pigz -d $data_dir/lid201-data-unsampled.tsv.gz
    else
        pigz -d $data_dir/lid201-data-unsampled.tsv.gz
    fi

    filter_african_languages $data_dir/lid201-data-unsampled.tsv $data_dir/lid201-data-unsampled-african.tsv

    python -m agbaye.lid.train_fasttext_model clean-dataset \
        --input_dataset $data_dir/lid201-data-unsampled-african.tsv \
        --output_dataset $data_dir/lid201-african-cleaned-includes-wura.tsv \
        --include_wura \
        --label_as_macrolanguage
}
