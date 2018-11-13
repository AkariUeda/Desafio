
# Desafio Movile

Neste desafio, estudamos o problema de classificação de spams em mensagens de SMS. 

Foram fornecidas pela Wavy amostras de mensagens de diferentes operadoras.


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', -1)

```


```python
dataset = pd.read_csv('SPAM Data _ Akari - SPAM Data.csv')
```


```python
print(len(dataset))
dataset.sample(frac=1).head(10)
```

    997





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor</th>
      <th>mensagem</th>
      <th>destino</th>
      <th>spam</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>619</th>
      <td>difize</td>
      <td>Oi FRANCISCO! Abasteca e pontue com seu cartao fidelidade na REDE CAXUXA e ganhe o DOBRO de pontos em sua primeira compra! Oferta valida por 30 dias.</td>
      <td>NaN</td>
      <td>False</td>
      <td>239</td>
    </tr>
    <tr>
      <th>950</th>
      <td>mailserr</td>
      <td>Traga seu numero para CLARO ganhe mais internet para navegar, Whastapp a vontade, ligacoes ILIMITADAS para todo Brasil. Rocell Digital  whastapp 55991282580</td>
      <td>CLARO</td>
      <td>False</td>
      <td>624</td>
    </tr>
    <tr>
      <th>118</th>
      <td>zootude</td>
      <td>Carlos, a CLARO ainda precisa de sua ligacao! Retorne ate o final do dia de hoje no 30038080 ou 08002088080 ou acesse Claropaguefacil.com.br</td>
      <td>NaN</td>
      <td>False</td>
      <td>133</td>
    </tr>
    <tr>
      <th>850</th>
      <td>difize</td>
      <td>REDE GMAXX: Oi JOSE! Preparamos uma oferta para voce! Abasteca na REDE GMAXX e ganhe o dobro de pontos em sua proxima compra! Valido 30 dias!</td>
      <td>NaN</td>
      <td>False</td>
      <td>233</td>
    </tr>
    <tr>
      <th>896</th>
      <td>centigen</td>
      <td>Sebastiana,mantenha seu plano da TIM em dia! Efetue o pagamento da divida.Evite a sua permanencia do debito,caso ja tenha pago, favor desconsiderar.</td>
      <td>NaN</td>
      <td>False</td>
      <td>104</td>
    </tr>
    <tr>
      <th>279</th>
      <td>centigen</td>
      <td>Franciele,mantenha seu plano da TIM em dia! Efetue o pagamento da divida.Evite a sua permanencia do debito,caso ja tenha pago, favor desconsiderar.</td>
      <td>NaN</td>
      <td>False</td>
      <td>163</td>
    </tr>
    <tr>
      <th>864</th>
      <td>centigen</td>
      <td>Renato,mantenha seu plano da TIM em dia! Efetue o pagamento da divida.Evite a sua permanencia do debito,caso ja tenha pago, favor desconsiderar.</td>
      <td>NaN</td>
      <td>False</td>
      <td>608</td>
    </tr>
    <tr>
      <th>225</th>
      <td>centigen</td>
      <td>Eliene,mantenha seu plano da TIM em dia! Efetue o pagamento da divida.Evite a sua permanencia do debito,caso ja tenha pago, favor desconsiderar.</td>
      <td>NaN</td>
      <td>False</td>
      <td>128</td>
    </tr>
    <tr>
      <th>112</th>
      <td>centigen</td>
      <td>Caio,mantenha seu plano da TIM em dia! Efetue o pagamento da divida.Evite a sua permanencia do debito,caso ja tenha pago, favor desconsiderar.</td>
      <td>NaN</td>
      <td>False</td>
      <td>215</td>
    </tr>
    <tr>
      <th>595</th>
      <td>difize</td>
      <td>Oi EMERSON! Abasteca e pontue com seu cartao fidelidade na REDE CAXUXA e ganhe o DOBRO de pontos em sua primeira compra! Oferta valida por 30 dias.</td>
      <td>NaN</td>
      <td>False</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>



Na amostra acima, podemos ver que temos 5 features:
* Vendor: Nome da empresa que enviou a mensagem
* Mensagem: texto da mensagem
* Destino: Operadora do destinatário da mensagem
* Spam: se foi classificada como spam (true) ou não (false)
* Total: Quantidade de cópias da mensagem enviadas

## Abordagens

Existem inúmeras formas de explorar o problema. Primeiro, devemos olhar bem as características dos nossos dados.


```python
dataset['spam'].value_counts()
```




    False    991
    True       6
    Name: spam, dtype: int64



Aqui temos um claro problema de _skewed classes_, em que uma classe (negativa) é muito mais predominante do que a outra (positiva). Casos assim não são triviais de serem solucionados. Vamos então primeiramente explorar os dados e analisar uma possível solução.

Podemos começar vendo o que temos nesses SMSs classificados como spam


```python
dataset.loc[dataset['spam'] == True]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor</th>
      <th>mensagem</th>
      <th>destino</th>
      <th>spam</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>772</th>
      <td>mailserr</td>
      <td>OI. Temos uma OFERTA especial de CELULAR ILIMITADO para todo BRASIL + 10GB de INTERNET. responda OK que retornarmos para voce ou ligue 0800 291 2253</td>
      <td>NaN</td>
      <td>True</td>
      <td>2276</td>
    </tr>
    <tr>
      <th>783</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Parabens! Seu chip esta ativado, realize uma ligacao de 30 segundos para confirmar o funcionamento. Digite se ja esta utilizando, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>495</td>
    </tr>
    <tr>
      <th>785</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Seu chip foi ativado e liberado para fazer ligacao. Utilize com urgencia, p confirmar o sinal! Digite 1 se ja esta utilizando, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>1041</td>
    </tr>
    <tr>
      <th>786</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Seu chip foi ativado e liberado para fazer ligacao. Utilize com urgencia, p confirmar o sinal! Digite 1 se ja esta utilizando, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>458</td>
    </tr>
    <tr>
      <th>787</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Seu chip ja foi ativado e esta gerando fatura. Digite 1 se ja realizou alguma ligacao com seu chip novo, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>283</td>
    </tr>
    <tr>
      <th>788</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Verificamos que voce ainda nao utilizou seu chip, e estamos gerando fatura! Faca uma ligacao com urgencia usando o seu chip da TIM.</td>
      <td>NaN</td>
      <td>True</td>
      <td>193</td>
    </tr>
  </tbody>
</table>
</div>



A primeira coisa que nos chamou a atenção foi a grande semelhança entre a classe positiva e negativa. Notei apenas erros sutis de ortografia ou gramática. Além disso, 5 dos 6 SMS classificados como spams foram enviados pela "quasiyo". Vamos verificar se todos os SMS enviados pela quasiyo são spams.


```python
dataset.loc[dataset['vendor'] == 'quasiyo'].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor</th>
      <th>mensagem</th>
      <th>destino</th>
      <th>spam</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>238</th>
      <td>quasiyo</td>
      <td>Esta difIcil lidar com as taxas de emprestimos? CLARO Q NAO! So RedeCifrao lhe apresenta a menor taxa e ainda diminui o valor da sua parcela! Resp. LIMITE (1/2)</td>
      <td>NaN</td>
      <td>False</td>
      <td>781</td>
    </tr>
    <tr>
      <th>239</th>
      <td>quasiyo</td>
      <td>Esta difIcil lidar com as taxas de emprestimos? CLARO Q NAO! So UBLA lhe apresenta a menor taxa e ainda diminui o valor da sua parcela! Resp. LIMITE p/+ info.</td>
      <td>NaN</td>
      <td>False</td>
      <td>1128</td>
    </tr>
    <tr>
      <th>240</th>
      <td>quasiyo</td>
      <td>Esta dificil lidar com as taxas de emprestimos? CLARO QUE NAO! So a REDE CIFRAO lhe apresenta a menor taxa e ainda diminui o valor da sua parcela! Resp. LIMIT</td>
      <td>NaN</td>
      <td>False</td>
      <td>1923</td>
    </tr>
    <tr>
      <th>241</th>
      <td>quasiyo</td>
      <td>Esta dificil lidar com as taxas de emprestimos? CLARO QUE NAO! So a REDE CIFRAO lhe apresenta a menor taxa e ainda diminui o valor da sua parcela! Resp. LIMITE</td>
      <td>NaN</td>
      <td>False</td>
      <td>186</td>
    </tr>
    <tr>
      <th>242</th>
      <td>quasiyo</td>
      <td>Esta dificil lidar com as taxas de emprestimos? CLARO QUE NAO! So a RedeCifrao lhe apresenta a menor taxa e ainda diminui o valor da sua parcela! Resp. LIMITE</td>
      <td>NaN</td>
      <td>False</td>
      <td>1067</td>
    </tr>
  </tbody>
</table>
</div>



Confirmamos que vários outros SMSs enviados pela quasiyo não foram classificados como spam. Será que existe algum padrão nas mensagens spams?


```python
dataset[dataset['mensagem'].str.contains("Ola, somos da TIM!")]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor</th>
      <th>mensagem</th>
      <th>destino</th>
      <th>spam</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>783</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Parabens! Seu chip esta ativado, realize uma ligacao de 30 segundos para confirmar o funcionamento. Digite se ja esta utilizando, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>495</td>
    </tr>
    <tr>
      <th>785</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Seu chip foi ativado e liberado para fazer ligacao. Utilize com urgencia, p confirmar o sinal! Digite 1 se ja esta utilizando, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>1041</td>
    </tr>
    <tr>
      <th>786</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Seu chip foi ativado e liberado para fazer ligacao. Utilize com urgencia, p confirmar o sinal! Digite 1 se ja esta utilizando, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>458</td>
    </tr>
    <tr>
      <th>787</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Seu chip ja foi ativado e esta gerando fatura. Digite 1 se ja realizou alguma ligacao com seu chip novo, 2 se nao.</td>
      <td>NaN</td>
      <td>True</td>
      <td>283</td>
    </tr>
    <tr>
      <th>788</th>
      <td>quasiyo</td>
      <td>Ola, somos da TIM! Verificamos que voce ainda nao utilizou seu chip, e estamos gerando fatura! Faca uma ligacao com urgencia usando o seu chip da TIM.</td>
      <td>NaN</td>
      <td>True</td>
      <td>193</td>
    </tr>
  </tbody>
</table>
</div>



Como apenas mensagens de spams tem o texto "Ola, somos da TIM!", uma solução seria classificar todas as mensagens com esse texto como spam. Mas claramente não seria um bom classificador. 

Com estes testes, concluímos que as mensagens spams e não spams são muito semelhantes. É muito complicado criar um modelo de aprendizado de máquina para fazer uma tarefa que nem mesmo nós, humanos, seríamos capazes de fazer.  

O único padrão que pudemos encontrar nos spams foi erros de ortografia e gramática. Provavelmente as mensagens foram classificadas como spams por este motivo. Outras features, como o total de cópias da mensagem, também não apresentaram nenhum padrão. Poderíamos propor filtrar as mensagens através de um corretor ortográfico, mas aparentemente é um padrão nas mensagens de SMS não utilizar acentuação, o que nos geraria muitos falsos positivos. 


```python

```


```python

```


```python

```


```python

```


```python

```

Podemos primeiramente fazer uma regressão, observar os resultados e, se necessário, melhorar a solução. Para isso, precisamos separar o conjunto em treino, validação e teste.


```python
label = dataset['spam']
data = dataset.drop('spam', axis=1)

#Separar em treino e teste aleatoriamente
X_train, X_test, y_train, y_test =  train_test_split(data, label, test_size=0.15, random_state=5)

#Separar em treino e validação
X_train, X_valid, y_train, y_valid =  train_test_split(X_train, y_train, test_size=0.3, random_state=5)
```


```python

```
