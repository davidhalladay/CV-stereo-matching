# Stereo Matching

## Environment_setting

- System : MacOS Mojave

- Python version : 3.7.4

- Opencv-contrib : 3.4.2

- Numpy : 1.16.2

- Matplotlib : 3.0.3

  

## Part_1

![79381004_2670840826342685_6687252775518076928_n](/Users/davidfan/Downloads/79381004_2670840826342685_6687252775518076928_n.jpg)

## Part_2

#### Explain your algorithm in terms of the standard 4-step pipeline. (cost computation, cost aggregation, disp. optimization, disp. refinement)

- cost computation : 我使用Census transform的方式，來計算L,R的cost，另外在左右邊界上，我們可能會有無法計算cost的情況，因為window已經超過整個圖片，因此我會將這些沒有辦法計算的邊界，沒有計算到的disparity設成最大，這樣就不會有錯誤的disparity計算。
##### Census transform 概念
![image-20191227102526437](/Users/davidfan/Library/Application Support/typora-user-images/image-20191227102526437.png)
-  cost aggregation : 這裡我使用blur來做均勻化的效果。
-  disp. optimization : 利用助教提供的方法，winner-take-all。
-   disp. refinement : 這裡我亦使用助教建議的方式，先做
	- Left-right consistency check，在左右兩個視角的深度圖上，找到不合理的點，再作對齊。
	
	- 再使用cv2.medianBlur，將生成出來的深度圖作均勻化處理。
	
	- 以及hole filling，將這些出來的圖形中，有小洞的地方補齊。
	
	- 最後使用bilateralFilter，進行最後的勻化處理。
	
	  


#### Show your output disparity maps in the report

| tsukuba                                                      | teddy                                                        | venus                                                        | cones                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![tsukuba](/Users/davidfan/Desktop/onedrive/大四(降三)/CV/hw4/CV2019_hw4/best/tsukuba.png) | ![teddy](/Users/davidfan/Desktop/onedrive/大四(降三)/CV/hw4/CV2019_hw4/best/teddy.png) | ![venus](/Users/davidfan/Desktop/onedrive/大四(降三)/CV/hw4/CV2019_hw4/best/venus.png) | ![cones](/Users/davidfan/Desktop/onedrive/大四(降三)/CV/hw4/CV2019_hw4/best/cones.png) |
| 1.44 sec                                                     | 2.56 sec                                                     | 7.52 sce                                                     | 7.78 sce                                                     |

#### Show your bad pixel ratio of each disparity maps in the report



![Screen Shot 2019-12-27 at 10.23.54 AM](/Users/davidfan/Desktop/Screen Shot 2019-12-27 at 10.23.54 AM.png)





#### Your reference papers or websites.

- Census Transform, https://blog.csdn.net/u010368556/article/details/72621099

- Hole Filling, https://blog.csdn.net/u012876599/article/details/51603033
- Hole Filling, https://blog.csdn.net/wc781708249/article/details/78539990
- Filter, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
- Box filter, http://tech-algorithm.com/articles/boxfiltering/
- Computing the Stereo Matching Cost with a Convolutional Neural Network, CVPR 2015,https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zbontar_Computing_the_Stereo_2015_CVPR_paper.pdf
- Left right consistency check, https://www.coursehero.com/file/p6anjr6/Left-right-consistency-check-For-each-pixel-p-l-of-the-left-view-Lookup-p-l-s/
- Weighted Median Filter, CVPR 2014, https://blog.csdn.net/streamchuanxi/article/details/79573302# -CV-stereo-matching
