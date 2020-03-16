---
marp: true
paginate: true
theme: gaia-ex
style: |
  section.first > h2 {
    font-size: 35px;
  }
---

<!-- _class: first -->

# 「ベイズ推論による機械学習」勉強会 (4)

## 第 3 章: ベイズ推論による学習と予測

### 正好 奏斗(@cosnomi)

---

<!-- _class: chapter -->

## ポアソン分布の学習と予測

---

- やるべきことは、前と同じ(実際の計算が違うだけ)なので、サクッと
- $p(x|\lambda) = \mathrm{Poi}(x|\lambda)$ (尤度分布)
- $\lambda \in \mathbb{R}^+$で、共役事前分布はガンマ分布
- $p(\lambda) = \mathrm{Gam}(\lambda | a,b)$とする (事前分布)

---

## 事後分布

- $X=\{x_1, \dots , x_N\}$を学習したとき、事後分布は？
- 計算省略
- $p(\lambda | X) = \mathrm{Gam}(\lambda | \hat{a}, \hat{b})$
- $\hat{a}=\sum_n x_n+a$
- $\hat{b}=\N+b$

---

## 予測分布

- $\int \mathrm{Poi}(x_*|\lambda) \mathrm{Gam}(\lambda | \hat{a}, \hat{b}) d\lambda$
- 計算省略
- 得られる分布は負の二項分布 $\mathrm{NB}(x_* | r, p)$
- $r=\hat{a}, p = \frac{1}{\hat{b}+1}$
- **NOTE**: 尤度分布と予測分布は同じではない
  - 尤度分布にポアソン分布を仮定したけど、それはパラメータ$\lambda$を固定したときの分布
  - 予測分布はパラメータについて積分(周辺化)するので、パラメータ自体の分布のせいで異なる形の分布になることはある

---

<!-- _class: chapter -->

## 1 次元ガウス分布の学習と予測

---

## 何を学習するのか

- ガウス分布にはパラメータが 2 つ ($\mu, \sigma^2$)あるので、何を学習するのかは 3 パターンある
  - データがガウス分布に従うと仮定できるけど、
  1. 平均は分からないが、分散は分かる
  2. 分散は分からないが、平均は分かる
  3. どちらも分からない
- NOTE: これ以降、数式の表記上、$\sigma^2$の代わりに、精度パラメータ$\lambda = \frac{1}{\sigma^2}$を用いる。$\lambda$が大きいほど、平均付近にまとまった分布になる

---

<!-- _class: chapter -->

## 平均のみが未知の場合

---

## 事前分布

- まずベースとなるガウス分布 $p(x|\mu) = N(x|\mu, \lambda^{-1})$
- やりたいこと: この$\mu$に事前分布を決め、いくつかサンプルした$x$(訓練データ)から、$x$の事後分布を求めたい
- 今回は、$p(\mu) = N(\mu | m, \lambda_{\mu}^{-1})$を事前分布とする
  - $m$と$\lambda_{\mu}^{-1}$はハイパーパラメータ
    - 尤度分布の$\lambda$とは違う
  - これは共役事前分布である
- 平均のみが未知のときはこれが共役になるが、推定したいパラメータが異なれば、事前分布も異なってくる

---

## 事後分布の計算

$p(\mu|X) \propto p(X | \mu) p(\mu) = \{\prod_n N(x_n | \mu, \lambda^{-1})\}N(\mu | m, \lambda_{\mu}^{-1})$

- ここからは正規分布の対数をとって計算する (省略します！)
- 結果として、次のようなパラメータをもつガウス分布を得る

  - $\hat{\lambda_\mu} = N\lambda + \lambda_{\mu}$
  - $\hat{m} = \frac{\lambda \sum_n x_n + \lambda_{\mu} m}{\hat{\lambda}_\mu}$

- 定性的に何が言える？
  - 平均の精度は N が増えるほど高くなる
  - $\hat{m}$は N が増えるほど、$\lambda \sum_n x_n$が重くなってくる ($\lambda_\mu m$より)

---

## 予測分布

$p(x_* | X) = \int p(x_* | \mu, \lambda^{-1})p(\mu | X) d\mu$

- 込み入った計算になるので省略
- 結果として得られる分布は$p(x_* | X) = N(x_* | \mu_*, \lambda_*^{-1})$
  - $\lambda_*=\frac{\lambda \hat{\lambda}_\mu}{\lambda + \hat{\lambda}_\mu}$
    - $N$が非常に大きければ、$\hat{\lambda}_\mu \approx N\lambda$
    - このとき、$\lambda_* = \lambda$ (精度はほぼ変わらないということ)
  - $\mu_* = \hat{m}$

---

## スピードアップ

![h:250px](./skate_speed_pursuit.png)
ここからは今までよりもさらに雑に追っていきます

なぜ？

- 詳しい計算なら本にしっかり載っている
- 早く MCMC 法あたりまで進んで実際にコードを書いていきたい

---

<!-- _class: chapter -->

## 精度のみが未知の場合

---

## 事前分布

- まずベースとなるガウス分布 $p(x|\lambda) = N(x|\mu, \lambda^{-1})$
- やりたいこと: この$\lambda$(← ここが違う)に事前分布を決め、いくつかサンプルした$x$(訓練データ)から、$x$の事後分布を求めたい
- 今回は、$p(\lambda)=\mathrm{Gam}(\lambda | a,b)$を事前分布とする
  - $a,b$はハイパーパラメータ
  - この分布は正の実数値を生成する ($\lambda$の性質を満たす)
  - これは共役事前分布である

---

- 事後分布 $p(\lambda | X) = \mathrm{Gam}(\lambda | \hat{a}, \hat{b})$

  - $\hat{a} = N/2 + a$, $\hat{b} = 1/2 \sum_n (x_n-\mu)^2 + b$

  ![br h:100px](./spiritual_woman.png)

- 予測分布
  - スチューデントの t 分布を用いる
    ![](./student_t.jpg)
    ![tr h:250px](student_t_graph.jpg)
  - 明らかに関わっちゃいけないタイプの式なんですが、これ、みなさん(私含む)レポートでよく使ってたんですよね。
  - これを用いて、$p(x_*) = \mathrm{St}(x_*|\mu_s, \lambda_s,\nu_s$)
  - $\mu_s=\mu, \lambda_s =a/b, \nu_s=2a$

---

<!-- _class: chapter -->

## 平均・精度が未知の場合

---

## 事前分布

- 尤度分布の設定 $p(x|\mu, \lambda) = N(x|\mu, \lambda^{-1})$
- 共役事前分布 $p(\mu, \lambda) = \mathrm{NG}(\mu, \lambda | m \beta, a, b)$
  - $\mathrm{NG}$はガウス・ガンマ分布
  - $N(\mu|m, (\beta \lambda)^{-1})\mathrm{Gam}(\lambda|a,b)$
- 今まで通り、計算は省略する部分が多いんですが、今回は予測する変数が 2 つ($\mu, \lambda$)なのは注意してみていきたいです。

- 平均パラメータが精度パラメータに依存している

---

![tr h:200px](graphical.jpg)

- まず、平均の方は$\lambda$を決めてしまえば、前と同じ
  ![](01.jpg)
- 基本に戻って、同時分布の構築
  - $p(X, \mu, \lambda) = p(\mu|\lambda, X) p(\lambda|X)p(X)$
    ![w:500px](03.jpg)
    ![br h:200px](02.jpg)

---

## 予測分布

$$p(x_* | X) = \iint p(x_* | \mu, \lambda) p(\mu, \lambda | X) \ d\mu d\lambda$$

- 結局、パラメータが取りうるすべての値について積分することは変わらない
  ![h:250px](04.jpg)

---

<!-- _class: chapter -->

## 線形回帰

---

- 「多次元ガウス分布に対する推論」は飛ばします
  - 計算を省略するなら、やることは一次元とほぼ同じなので
  - 追うのが辛い、時間がない
- 線形回帰は入力と出力がある「いわゆる AI っぽい」やつです
  - すごく語弊がありますが
- $y_n = \bm{w}^T \bm{x}_n + \epsilon_n$
- $\epsilon_n$は誤差を表す
  - 全てのデータが綺麗に乗っかる直線を見つけるのは無理なので
- $\epsilon_n \propto N(\epsilon_n | 0, \lambda^{-1})$
  - この$\lambda$はハイパーパラメータとします

---

- つまり、$p(y_n|\bm{x_n}, \bm{w}) = N(y_n | w^T \bm{x}\_n, \lambda^{-1})$
- この式の$\bm{w}$が今回学習したいパラメータ
