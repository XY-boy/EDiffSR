# EDiffSR (IEEE TGRS 2024)
### 📖[**Paper**](https://ieeexplore.ieee.org/document/10353979) | 🖼️[**PDF**](./img/EDiffSR.pdf)

PyTorch codes for "[EDiffSR: An Efficient Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/document/10353979)", **IEEE Transactions on Geoscience and Remote Sensing**, 2024.

- Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Kui Jiang](http://homepage.hit.edu.cn/jiangkui), [Jiang He](https://jianghe96.github.io/), Xianyu Jin, and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
- Wuhan University and Harbin Institute of Technology
 
## 🧩Usage
### Dataset Preparation
**Step I.** Please download the following remote sensing benchmarks:
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | [DIOR](https://www.sciencedirect.com/science/article/pii/S0924271619302825) | [NWPU-RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544)
| :----: | :-----: | :----: | :----: | :----: |
|Training | [Download](https://onedrive.live.com/?authkey=%21AAqO0B6SeejPkr0&id=42EC9A19F3DE58D8%2176404&cid=42EC9A19F3DE58D8&parId=root&parQt=sharedby&o=OneUp) | None | None | None |
|Testing | [Download](https://captain-whu.github.io/AID/) | [Download](https://captain-whu.github.io/DOTA/dataset.html) | [Download](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | [Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp)

**Step II.** Modify the path in `options/train/setting.yml` and `options/test/aid.yml`

### Train
```
python train.py -opt=options/train/setting.yml
```

### Test
```
python test.py -opt=options/test/aid.yml
```

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com

## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support！😊

```
@ARTICLE{xiao2024ediffsr,
  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and He, Jiang and Jin, Xianyu and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={EDiffSR: An Efficient Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2023.3341437}
}
```

