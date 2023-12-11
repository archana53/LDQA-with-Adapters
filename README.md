# LDQA-with-CrossAttention
Long Document Question Answering with Cross-Attention Grounding


## Datasets

### TweetQA

Train statistics:                                                                                                    
|                 |  Min  |  Max  |        Mean        | Median | 90th percentile | 95th percentile | 99th percentile |
| :-------------- | :---: | :---: | :----------------: | :----: | :-------------: | :-------------: | :-------------: |
| Query length    |   6   |  47   | 11.109333992004395 |  11.0  |      15.0       |      17.0       |      20.0       |
| Document length |  12   |  91   | 41.494014739990234 |  42.0  |      51.0       |      54.0       |      64.0       |
| Label length    |   3   |  28   | 5.866255283355713  |  5.0   |       9.0       |      10.0       |      13.0       |

Validation statistics:                                                                                               
|                 |  Min  |  Max  |        Mean        | Median | 90th percentile | 95th percentile |  99th percentile  |
| :-------------- | :---: | :---: | :----------------: | :----: | :-------------: | :-------------: | :---------------: |
| Query length    |   6   |  31   | 10.525782585144043 |  10.0  |      14.0       |      16.0       |       21.0        |
| Document length |   3   |  79   | 41.83241271972656  |  42.0  |      53.0       |      57.0       | 74.15000000000009 |
| Label length    |   3   |  16   | 5.581031322479248  |  5.0   |       8.0       |       9.0       |       12.0        |

### SQuAD

Train statistics:                                                                                                  
|                 |  Min  |  Max  |        Mean        | Median | 90th percentile | 95th percentile | 99th percentile |
| :-------------- | :---: | :---: | :----------------: | :----: | :-------------: | :-------------: | :-------------: |
| Query length    |   3   |  63   | 14.350666046142578 |  14.0  |      20.0       |      22.0       |      27.0       |
| Document length |  28   |  880  | 158.94517517089844 | 146.0  |      243.0      |      281.0      |      375.0      |
| Label length    |   3   |  79   | 6.654995918273926  |  5.0   |      11.0       |      15.0       |      26.0       |
                                                                                
Validation statistics:                                                                                             
|                 |  Min  |  Max  |        Mean        | Median | 90th percentile | 95th percentile | 99th percentile |
| :-------------- | :---: | :---: | :----------------: | :----: | :-------------: | :-------------: | :-------------: |
| Query length    |   6   |  42   | 14.493660926818848 |  14.0  |      20.0       |      22.0       |      27.0       |
| Document length |  31   |  785  | 162.8765411376953  | 148.0  |      248.0      |      283.0      |      429.0      |
| Label length    |   3   |  43   | 6.418542861938477  |  5.0   |      11.0       |      14.0       |      23.0       |