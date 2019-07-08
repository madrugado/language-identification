
Example data for problem 1:

```
This is an English sentence
Esta es una frase en español
Isto é uma frase em português
```

Example output for the problem 1:
```
en
es
pt
```

Example data for the problem 2:

```
Tem gente no Barcelona que vê Philippe Coutinho como um possível substituto de Neymar. 
Belga estará perto de renovar, mas, enquanto tal não sucede, o Chelsea teme que este mude de ideias e rume ao Real Madrid.
```

Example output for the problem 2:
```
pt-br
pt-pt
```



## Training data

Due to excessive size of langid data it is placed in the cloud: https://yadi.sk/d/jyAux1wTTlcRqA

Inside the data folder you will find data for the languages we are focusing on. Each is a monolingual file of a particular language and a language variant:

* EN - data.en
* ES - data.es
* PT-PT - data.pt-pt
* PT-BR - data.pt-br

Each dataset has also an additional .ids file which has the origin of the data. The data is in plain UTF-8 text format.
