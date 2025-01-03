First, clone the official repository of AIO
```
git clone https://github.com/cl-tohoku/AIO3_FiD_baseline
```
Note
- No need for docker in transfer2
- Download our trained models (official model release from AIO has ended)
# preparation for retriver
- download retriver and embedding from [Google Drive](https://drive.google.com/file/d/1NIEOgGuHe6eo2PSYR-EnrCT4q_W92tDI/view?usp=sharing)
- ``mkdir retrivers/AIO3_DPR/model``
- put them into the above directory

# preparation for FiD
- download FiD from [Google Drive](https://drive.google.com/file/d/16SSqpPLZ6QDoGFg8u76f_H9sBlDJ30Ie/view?usp=sharing)
- ``mkdir generators/fusion_in_decoder/model``
- put them into the above directory 

# run retriever
```
cd retrievers/AIO3_DPR
bash scripts/download_data.sh datasets
bash scripts/retriever/retrieve_passage.sh -n baseline -m model/baseline/retriever/dpr_biencoder.59.pt -e model/baseline/embeddings/emb_dpr_biencoder.59.pickle
cd -
```

# run generator(FiD)
```
python prepro/convert_dataset.py DprRetrieved fusion_in_decoder

cd generators/fusion_in_decoder
bash scripts/test_generator.sh configs/test_generator_slud.yml
cd -
```
