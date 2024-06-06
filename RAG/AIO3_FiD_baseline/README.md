clone the official repository of AIO
```
git clone https://github.com/cl-tohoku/AIO3_FiD_baseline
```
- No need for docker in transfer2
- Download our trained models (official model release from AIO has ended)
# preparation for retriver
- download retriver and embedding from [dropbox](https://www.dropbox.com/scl/fi/rayemlpb4jiqfnsuxjmpy/retrivers.tar.gz?rlkey=637yjm7npn0js1gi3hlo0zjxi&st=qmlsl3hl&dl=0)
- mkdir retrivers/AIO3_DPR/model
- put them into the above directory

# preparation for FiD
- download FiD from [dropbox](https://www.dropbox.com/scl/fi/woo1gm5kub01mvmw1i8yl/fusion-in-decoder.tar.gz?rlkey=u3lgx2ql604pbqz0zq9j6b6z5&st=fcyn6mzl&dl=0)
- mkdir generators/fusion_in_decoder/model
- put them into the above directory 

```
cd retrievers/AIO3_DPR
bash scripts/download_data.sh datasets
bash scripts/retriever/retrieve_passage.sh -n baseline -m model/baseline/retriever/dpr_biencoder.59.pt -e model/baseline/embeddings/emb_dpr_biencoder.59.pickle
cd -

python prepro/convert_dataset.py DprRetrieved fusion_in_decoder

cd generators/fusion_in_decoder
bash scripts/test_generator.sh configs/test_generator_slud.yml
cd -
```
