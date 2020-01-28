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

- まず確率分布に対して定義される値(期待値)を学びます
- 離散・連続型確率分布について、その定義と性質を見ていきます
  - 多いです

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
- スカラーなら「平均との差(=残差)の2乗の平均」
- <strong>分散</strong>の定義
  - $\langle  (\bm{x}-\langle \bm{x} \rangle)(\bm{x}-\langle \bm{x} \rangle)^T \rangle$
  - 先程の2乗の平均に相当する値$\langle \bm{x}\bm{x}^T\rangle_{p(\bm{x})}$を思い出す
  - $\bm{x}-\langle \bm{x} \rangle$が残差なので結局、残差2乗平均っぽい
$$\begin{aligned} \langle \bm{x}\bm{x}^T-\bm{x}\langle \bm{x}^T\rangle-\langle \bm{x} \rangle \bm{x}^T + \langle \bm{x} \rangle\langle \bm{x} \rangle^T \rangle \\ = \langle \bm{x}\bm{x}^T \rangle - \langle \bm{x} \rangle\langle \bm{x} \rangle^T  \end{aligned}$$
- 