# dehaze 暗通道去雾
Realizing Kaiming He's paper 'Single Image Haze Removal Using Dark Channel Prior'

- 雾霾天一定是白天，天色灰蒙蒙的。因此在计算大气光时，我假设最亮部分是灰白色，那RGB应该都很接近255，so没必要分开算R/G/B，具体见代码
- When I calculate the A(atmosphere), I simply suppose the brightest spot is approximately white(gray). So the RGB value of that point should be almost same
- 使用导向滤波精确透光率
- Using the guided filter now.

