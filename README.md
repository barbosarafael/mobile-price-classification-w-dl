# Classificação de preço de telefone com Deep Learning

## 1) Contexto pessoal

Esse é um projeto pessoal para estudar sobre conceitos em Deep Learning. Meu foco será em testar o `pytorch` e aplicá-lo no dataset para classificação.

## 2) Dados

### 2.1) Contexto do projeto

Os dados foram retirados do [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) e o objetivo do projeto é estimar o preço dos aparelhos de telefone. Para isso, o preço foi codificado em 4 faixas:

- 0: custo baixo
- 1: custo médio
- 2: custo alto
- 3: custo muito alto

Ainda no [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification), o autor contextualiza um pouco mais o problema:

> Bob has started his own mobile company. He wants to give tough fight to big companies like Apple, Samsung etc. He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones from various companies. Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem. In this problem you do not have to predict actual price but a price range indicating how high the price is

### 2.2) Dicionário dos dados

1. **Id**
2. **battery_power**: Total energy a battery can store in one time measured in mAh
3. **blue**: Has bluetooth or not
4. **clock_speed**: speed at which microprocessor executes instructions
5. **dual_sim**: Has dual sim support or not
6. **fc**: Front Camera mega pixels
7. **four_g**: Has 4G or not
8. **int_memory**: Internal Memory in Gigabytes
9. **m_dep**: Mobile Depth in cm
10. **mobile_wt**: Weight of mobile phone
11. **n_cores**: Number of cores of processor
12. **pc**: Primary Camera mega pixels
13. **px_height**: Pixel Resolution Height
14. **px_width**: Pixel Resolution Width
15. **ram**: Random Access Memory in Megabytes
16. **sc_h**: Screen Height of mobile in cm
17. **sc_w**: Screen Width of mobile in cm
18. **talk_time**: longest time that a single battery charge will last when you are
19. **thee_g**: Has 3G or not
20. **touch_screen**: Has touch screen or not
21. **wifi**: Has wifi or not

### 2.2) Extração dos dados

Criei esse script `.py` que extrai os dados desse dataset no Kaggle e salva em uma pasta chamada `data`

```
python get_dataset.py -p iabhishekofficial/mobile-price-classification
```

## 3) 