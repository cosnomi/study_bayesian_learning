---
marp: true
paginate: true
theme: gaia-ex
---

<!-- _class: first -->

# 「ベイズ推論による機械学習」勉強会

## 第 2 章: 基本的な確率分布

### 正好 奏斗(@cosnomi)

---

## この章について

- まず確率分布に対して定義される値(期待値など)を学びます
- 離散・連続型確率分布について、その定義と性質を見ていきます
  - 多いです
  - 各分布における期待値などを ~~一応導出します~~(長すぎるので一部省略) が、実務上は計算済みの表を見たほうが速いと思います
  - 実際の式・計算より「**分布が何を表す(いつ使える)のか**」「分布の**パラメータ**は何か」が大事
    - 今後、推論に入っていくときに使う

---

## 目次1

<!-- _class: outline -->

1. 確率分布に関連する値
2. 離散確率分布
   1. ベルヌーイ分布
   2. 二項分布
   3. カテゴリ分布
   4. 多項分布
   5. 4つの分布の関係
   6. ポアソン分布

---

## 目次2

<!-- _class: outline -->

3. 連続型確率分布
   1. ベータ分布
   2. ディレクレ分布
   3. ガンマ分布
   4. 1次元ガウス分布(正規分布)
   5. 多次元ガウス分布
   6. ウィシャート分布

---

## 期待値

- 定義
  - $\bm{x}$をベクトル、$p(\bm{x})$を確率分布、$f(\bm{x})$を$\bm{x}$に対して定義される関数とする。$p(\bm{x})$に対する$f(\bm{x})$の期待値は
$$ \langle f(\bm{x}) \rangle _{p(\bm{x})} = \int f(\bm{x})p(\bm{x}) dx $$

- ここでは$\bm{x}$をベクトルとしているけど、スカラーのほうが身近
- 右下の$p(\bm{x})$は考えている分布が明らかなときは省略 $\langle f(\bm{x}) \rangle$
- <strong>線形性</strong>: $\langle af(\bm{x})+bg(\bm{x}) \rangle=a\langle f(\bm{x}) \rangle+b\langle g(\bm{x}) \rangle$

---

## 平均
- $f(\bm{x}) = \bm{x}$としたときの期待値
  - つまり $\langle \bm{x} \rangle_{p(\bm{x})}$
- 期待値の定義より
$$ \langle \bm{x} \rangle _{p(\bm{x})} = \int \bm{x} p(\bm{x}) dx $$
- 確かに平均っぽい

- <strong>2乗の平均</strong>に相当する値: $\langle \bm{x}\bm{x}^T\rangle_{p(\bm{x})}$
  - スカラーで考えると、$\langle x^2 \rangle_{p(\bm{x})}$
---

## 分散
- スカラーなら「平均との差(=残差)の2乗の平均」であった
- **分散**の定義 $\langle(\bm{x}-\langle \bm{x} \rangle)(\bm{x} - \langle \bm{x} \rangle)^T \rangle$
  - 先程の2乗の平均に相当する値$\langle \bm{x}\bm{x}^T\rangle_{p(\bm{x})}$を思い出す
  - $\bm{x}-\langle \bm{x} \rangle$が残差なので結局、残差2乗平均っぽい
    $$\begin{aligned} \langle \bm{x}\bm{x}^T-\bm{x}\langle \bm{x}^T\rangle-\langle \bm{x} \rangle \bm{x}^T + \langle \bm{x} \rangle\langle \bm{x} \rangle^T \rangle \\ = \langle \bm{x}\bm{x}^T \rangle - \langle \bm{x} \rangle\langle \bm{x} \rangle^T  \end{aligned}$$
  - 2乗の平均-平均の2乗と同じ形

---

## 条件付き期待値
- $\langle\bm{x}\bm{y}^T\rangle_{p(\bm{x}, \bm{y})}$を考えたい
- 独立なら単に分解できて、$\langle\bm{x}\rangle_{p(\bm{x})} \langle\bm{y}^T\rangle_{p(\bm{y})}$
- 独立でないとき $\langle\bm{x}\bm{y}^T\rangle_{p(\bm{x}, \bm{y})} = \langle\langle\bm{x}\rangle_{p(\bm{x}|\bm{y})}\bm{y}^T\rangle_{p(\bm{y})}$
  - 見た目は怖いけど、よく考えると妥当に見える
  - $\bm{x}$と$\bm{y}$の間には関係がある(←独立でない)から、外側の<>で$\bm{y}$を仮定してるのなら、単に$p(\bm{x})$ではなく$p(\bm{x}|\bm{y})$を考えてあげないといけない
  - $\langle\bm{x}\rangle_{p(\bm{x}|\bm{y})}$を**条件付き期待値**という

---

## エントロピー

- <span class="definition">**エントロピー**とは確率分布に対して定義される値で、</span>
  $$ \mathrm{H}[p(\bm{x})] := - \int p(\bm{x}) \ln p(\bm{x}) d\bm{x} = -\langle \ln p(\bm{x})\rangle_p(\bm{x}) $$
- 確率分布の「**乱雑さ**」を表す
  - 実は対数が大事なのではなく、$-\ln p(x) = \ln \frac{1}{p(x)}$で逆数を取っていることが「乱雑さ」を表現している
  - $p$は「起こりやすさ」、$1/p$は「予測しにくさ」という気分
- 定義なので受け入れるしかない…

<!-- TODO: 実際の計算？ -->

---

## KLダイバージェンス
- <span class="definition" /> 2つの確率分布$p(\bm{x}), q(\bm{x})$がどれくらい離れているかを表す値
  $$\begin{aligned} \mathrm{KL}[q(\bm{x}) || p(\bm{x})] &= -\int q(\bm{x})\ln \frac{p(\bm{x})}{q(\bm{x})} d\bm{x} \\ &= - \langle \ln p(\bm{x})\rangle_{q(\bm{x})} + \langle \ln q(\bm{x})\rangle_{q(\bm{x})}  \end{aligned}$$

- **注意**: $\mathrm{KL}[q(\bm{x}) || p(\bm{x})] \neq \mathrm{KL}[p(\bm{x}) || q(\bm{x})]$

---

## 離散確率分布
- ここから色々な離散確率分布を見ていきます。
- 必要な事前知識はこれだけ
  - 離散分布において、$p(\bm{x})$は$P(\bm{X}=\bm{x})$を表す
  - すなわち確率変数$\bm{X}$が$\bm{x}$である確率
  - $p(x)$は「サイコロの目がxである確率」みたいな感じ
- 「定義」「どんな意味を持つか」「期待値などの計算」を抑える
---

## ベルヌーイ分布の定義
- <span class="definition" />$x\in \{0,1\}$とパラメータ$\mu$を用いて
    $$\mathrm{Bern}(x|\mu) = \mu^x(1-\mu)^{1-x}$$
- この分布の気持ちは簡単
  - $x$は必ず0と1のどちらかを取る値
  - $x=1$となる確率が$\mu$となるような分布が欲しい
  - 定義の式にx=0とx=1を代入してみよう
- $\langle x \rangle = \mu$ 理由: $x=1$となる確率は$\mu$なので
- $\langle x^2 \rangle = \mu$ 理由: $0^2=0, 1^2=1$なので結局変わらない

---

## ベルヌーイ分布のエントロピー

![w:1000px](bern_entropy.jpg)
- 定義に従って計算しているだけ $\mathrm{H}[\mathrm{Bern}(x|\mu)] = -\langle \ln p(\bm{x}) \rangle$
- 確率変数でない定数は外に出せる ($\mu$とか)

---

## ベルヌーイ分布のKLダイバージェンス

- パラメータの異なる2つのベルヌーイ分布のKLダイバージェンス
- 単なる計算練習(だと思ってます)
$$\begin{aligned} \mathrm{KL}[\mathrm{Bern}(x | \hat{\mu}) || \mathrm{Bern}(x | \mu)] &= - \langle \ln \mathrm{Bern}(x | \mu)\rangle_{\mathrm{Bern}(x | \hat{\mu})} + \langle \ln \mathrm{Bern}(x | \hat{\mu}))\rangle_{\mathrm{Bern}(x | \hat{\mu})} \\ &= -(\hat{\mu}\ln \mu + (1-\hat{\mu})\ln(1-\mu)) -\mathrm{H}[\mathrm{Bern}(x | \hat{\mu})] \\ &=...\end{aligned}$$
- エントロピーは既に求めているので代入して整理するだけ
  - この式ではエントロピーを求めた対象の分布のパラメータが$\hat{\mu}$であることに注意

---

## 二項分布

- 何を考えたいか
  - 1回投げると表が$p$の確率で出るコイン
  - **$M$回投げたとき、表が出た回数** $m \in \{0,1,...,M\}$の分布を考えたい ($M=1$ならベルヌーイ分布と同じ)
- <span class="definition" /> $\mathrm{Bin}(m|M,\mu) = {_MC_m} \mu^m (1-\mu)^{M-m}$
- $\langle m \rangle = \sum_{m=1}^M m \cdot {_MC_m} \mu^m (1-\mu)^{M-m} = ... = M\mu$
- $\langle m^2 \rangle = \sum_{m=1}^M m^2 \cdot {_MC_m} \mu^m (1-\mu)^{M-m} = ... = M\mu$
- 詳しく追いたい人向け: https://to-kei.net/distribution/binomial-distribution/b-parameter-derivation/

---

## 二項分布の形
- 細かい計算はいいとして…形を知っておくことは大事
![h:500 center](binominal_graph.jpg)

---

## one-hot表現
- ここからはコインの表裏のように$x \in \{0,1\}$ではなく、3つ以上の結果について考えます
- では、$x \in \{0,1,2,...,N\}$とすれば良いでしょうか
  - 0,1,2…には大小関係が定義される
  - 事象同士に**大小関係**を考えるべきか？
    - **名義尺度**: 量的な意味を全く持たない。質的な値。
  - それより各事象が0(起こらない), 1(起こる)を考えたい
- K個の事象が起こるとき、**K次元ベクトル**$\bm{s}$を考える
  - Kコの成分のうち**1つだけ1、それ以外は0** 例) $(0,1,0,0)^T$

---

## カテゴリ分布の定義

- ベルヌーイ分布を表裏だけでなく、**K次元**に拡張したい
  - サイコロなら$K=6$
  - 試行回数は**1回** (二項分布と混同しないで)
- K次元ベクトル$\bm{s}$と、それぞれの事象の確率$\pi_k (k=1,..,K)$

- <span class="definition"> $\mathrm{Cat}(\bm{s}|\pi) = \prod_{k=1}^K \pi_k^{s_k}$
- 欲しい確率以外は0乗されるので影響しない

---

## カテゴリ分布とベルヌーイ分布
$$\mathrm{Bern}(x|\mu) = \mu^x(1-\mu)^{1-x}, \ \ \  \mathrm{Cat}(\bm{s}|\pi) = \prod_{k=1}^K \pi_k^{s_k}$$
- 形が違うように見えるけど…?
  - 表or裏をカテゴリ分布で考えると$K=2$になる
  - ベルヌーイ分布では、$x=1$の確率が$\mu$なら、$x=0$の確率は1から引くだけ $1-\mu$という性質を利用して2次元を1次元にした
  - カテゴリ分布も$\pi_k=1-\sum_{n=1}^K \pi_n$を利用すれば、K-1次元に減らせるが、複雑なのでそうしていないだけ

---

## カテゴリ分布の計算
- 今、確率変数にスカラーではなくベクトルを考えているので、期待などもベクトル $\langle \bm{s} \rangle$になるが、各成分$s_k$について簡単に計算できる
- $\langle s_k \rangle = \pi_k$
- $\langle s_k^2 \rangle = \pi_k$ (結局、各成分は0or1だから2乗しても同じ)
![h:300px](cat_entropy.jpg)

---

## 多項分布の気持ち
- 多次元 + 多試行
- ![](multinominal_expansion.jpg)
- サイコロ($K=6$)を$M$回投げたとき$k$の目が$m$回出る確率は？

---

## 多項分布の定義
- K個の事象が起こる試行をM回行う
- $\bm{m}$はK次元ベクトルで、k番目の事象が起こった回数
- $\bm{\pi}$はK次元ベクトルで、1回の試行でk番目の事象が起こる確率
- <span class="definition"> 
$$\mathrm{Mult}(\bm{m}|\bm{\pi}, M) = M! \prod_{k=1}^K \frac{\pi_k^{m_k}}{m_k!}$$
- $M!$とか$m_k!$とかは「同じものがあるときの順列公式」による