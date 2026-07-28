# P1 — RAG + agente de produção + harness
 de avaliação (guia executável)
**Objetivo
:** um sistema RAG de nível produção sobre
 um corpus público, com **modo agente** (too
l use, HITL), uma **suíte de avaliação rig
orosa**, **gate de CI** e uma camada de **gov
ernança**. Prova profundidade em GenAI + **e
vals** (seu maior gap) + governança + LLMOps
. **Dupla função:** é a base do mestrado.

**Nota:** este é o flagship de IA do portfó
lio revisado — absorve o antigo "P2 (agente
 text-to-SQL)" (vira o modo agente aqui) e o 
antigo "P5 (harness LLMOps)" (é a suíte de 
evals + gate de CI aqui). Roda idealmente sob
re o lakehouse do P2 (fundação compartilhad
a).

> Frase-alvo de entrevista ao fim: *"Con
struí um RAG com harness de avaliação. Bas
eline naive deu X de groundedness; hybrid + r
eranking subiu pra Y (tenho a ablação). Gat
ei no CI pra regressão não subir. Adicionei
 retrieval com controle de acesso, mascaramen
to de PII e abstenção, e medi resistência 
a prompt injection."*

---

## Ancoragem no O
pen-Finance-LakeHouse (recomendado — compos
ição)
Em vez de um projeto solto, **ancore 
o P1 no seu P2** (o lakehouse que você já t
em). Isso amarra a narrativa *"eu construo a 
fundação E coloco IA governada em cima"*:
-
 **Corpus do RAG:** documentos/relatórios do
 domínio financeiro que o lakehouse já cobr
e (BACEN, notas de política monetária, rela
tórios ANBIMA/Tesouro) — PT + regulatório
, exercita embeddings multilíngues e o ângu
lo de governança que é a sua marca.
- **Mod
o agente:** um agente que faz **text-to-SQL s
obre os marts gold** (DuckDB) do lakehouse �
� responde perguntas de negócio consultando 
os dados reais que você já modelou, com too
l use, HITL e as mesmas evals/guardrails.
- *
*Governança compartilhada:** reaproveita a l
inhagem (OpenLineage/OpenMetadata) e o contro
le de acesso do lakehouse na camada de retrie
val.

## Corpus (alternativa standalone)
Se p
referir não ancorar: **resoluções da ANATE
L** (telecom, PT) ou normas de real estate. P
ara rigor de mestrado, rode **também** contr
a um dataset de QA público conhecido, para r
eprodutibilidade/comparação.

## Stack (100
% open-source e barato — regra do portfóli
o)
Tudo aberto, roda local, **~R$0**. Nada de
 API paga por padrão.
| Camada | Escolha (ab
erta) | Nota |
|---|---|---|
| Linguagem | Py
thon | — |
| Orquestração | **LangGraph**
 | aberto; você já lista |
| Vector store |
 **Qdrant** (self-host Docker) | aberto, grá
tis |
| Embeddings | **bge-m3** (HuggingFace,
 local) | aberto, multilíngue (PT), grátis 
|
| LLM | **Ollama local** (Llama 3.1/3.2, Qw
en2.5) | **grátis, sem vendor lock-in**; API
 paga é opcional |
| Reranker | **bge-rerank
er** (local) | aberto, grátis (evitar Cohere
, que é pago) |
| Structured output | **Pyda
ntic / Instructor** | aberto; você já lista
 |
| Avaliação | **Ragas** apontando pro mo
delo local + harness próprio | aberto; o cor
ação do projeto |
| PII | **Presidio** (Mic
rosoft, open-source) | grátis |
| Tracing/ob
s | **Langfuse** (self-host Docker) | aberto,
 grátis |
| Guardrails | NeMo Guardrails (ab
erto) ou validação custom | grátis |
| Ser
ving | **FastAPI** + **Streamlit** | aberto |

| CI | **GitHub Actions** (repo público) | 
grátis |

### Custo (aberto e barato)
- **~R
$0.** Ollama + bge-m3 + bge-reranker rodam **
no seu laptop** (CPU serve; GPU acelera).
- P
recisa de GPU? **Colab / Kaggle** dão GPU gr
átis. HuggingFace Spaces hospeda demo gráti
s.
- **Corpus ANATEL** é público. GitHub Ac
tions é grátis em **repo público**.
- Úni
co custo opcional: rodar UM comparativo contr
a um modelo frontier (GPT-4) pra referência 
de qualidade — dá pra usar créditos grát
is ou simplesmente pular. O default é tudo l
ocal.
- **Bônus de narrativa:** "rodei em mo
delos abertos, sem vendor lock-in" é um pont
o forte, não uma limitação.

## Estrutura 
do repo
```
rag-eval/
  README.md            
     # arquitetura, decisões, números antes
/depois
  docker-compose.yml        # qdrant 
+ langfuse + app
  ingest/                   
# loading, chunking, embedding do corpus
  re
trieval/                # hybrid (BM25+dense)
, reranking, filtros de metadado
  generation
/               # prompt, chamada LLM, citaç
ões, structured output
  guardrails/        
       # validação I/O, PII, prompt-injecti
on, abstenção
  eval/
    datasets/        
       # gold QA set VERSIONADO (o ativo cien
tífico)
    metrics/                # retrie
val + generation metrics
    run_eval.py     
        # gera relatório
    regression_gate
.py      # falha o CI se piorar
  serving/   
               # FastAPI + UI mínima
  gover
nance/               # controle de acesso no 
retrieval, classificação, audit log
  .gith
ub/workflows/eval.yml
  docs/writeup.md      
     # o artefato de entrevista
```

---

## 
Milestones (ordem executável)

**M0 — Scaf
folding (1 fim de semana).** Repo + `docker-c
ompose` (Qdrant + Langfuse) + ingestão de um
a fatia pequena do corpus. Objetivo: rodar po
nta a ponta.

**M1 — RAG baseline naive.** 
chunk → embed → top-k → stuff → respo
sta **com citações**. Fiar tracing no Langf
use. Já responde perguntas. *(Não otimize n
ada ainda — é o baseline que você vai bat
er.)*

**M2 — Gold eval set (o crux cientí
fico).** 50-100 pares Q/A com **spans de orig
em** (qual trecho responde). Versionado. **Es
se dataset é a ciência do projeto** — sem
 ele não há avaliação séria.

**M3 — H
arness de avaliação.** Roda o baseline e pr
oduz um **relatório** com:
- *Retrieval:* re
call@k, precision@k, MRR, nDCG (o trecho cert
o é recuperado?).
- *Geração:* faithfulnes
s/groundedness, answer relevance, context pre
cision/recall (via Ragas).
- *Ponta a ponta:*
 task success, correção de citação.
- *LL
M-as-judge:* rubrica + **calibração contra 
~30 rótulos humanos** (reporta concordância
; não confia cego no juiz — rigor de mestr
ado).

**M4 — Melhorar retrieval e MEDIR ca
da mudança.** hybrid (BM25 + denso), reranki
ng, estratégias de chunking, filtros de meta
dado. **Cada mudança gera um número de delt
a** no harness. *Aqui nasce a história:* "su
bi groundedness de X pra Y adicionando rerank
ing". Isso é a **ablação** — e é o núc
leo da dissertação.

**M5 — Guardrails + 
governança.** detecção/mascaramento de PII
; defesa a prompt-injection; **controle de ac
esso no retrieval** (usuário só recupera do
c que pode ver); **audit log** (quem pergunto
u o quê, quais fontes); **abstenção** quan
do a evidência é insuficiente. Métricas: t
axa de sucesso de ataque de injection, taxa d
e vazamento de PII, correção da abstenção
.

**M6 — Gate de CI (vira o P5).** GitHub 
Action roda o eval em cada PR e **barra regre
ssão**. Prova disciplina de LLMOps.

**M7 �
� Serving + demo.** FastAPI + UI mínima + pa
inel de custo/latência (do Langfuse).

**M8 
— Writeup.** arquitetura, decisões, númer
os antes/depois, modos de falha, custo. **É 
o artefato que você leva pra entrevista.**


---

## A profundidade de evals (o diferencia
dor — não pule)
O que separa "sei RAG" de 
"sei operar RAG":
- **Retrieval vs geração 
medidos separadamente** (a maioria mistura e 
erra o diagnóstico).
- **Ablação** — pro
var que cada componente (reranking, hybrid, c
hunk size) move uma métrica.
- **LLM-as-judg
e calibrado** — reportar concordância com 
humano, reconhecer o limite do juiz.
- **Adve
rsarial** — injection, perguntas fora de es
copo (deve **abster**), perguntas ambíguas.

- **Unit economics** — tokens, R$/query, p9
5 de latência. Trade-off qualidade × custo.


## Ângulo de mestrado
**Pergunta de pesqui
sa:** *como estratégia de retrieval, chunkin
g e reranking trocam qualidade × custo/latê
ncia, e como avaliar + governar RAG sobre dad
os corporativos sensíveis?* A ablação + o 
benchmark reprodutível são o miolo da disse
rtação. Entrega quádrupla: **dissertação
 + benchmark reproduzível + repositório pú
blico + palestra/artigo**.

## Escopo mínimo
 vs completo
- **MVP forte de portfólio:** M
0–M4 (RAG + harness real + ablação). Já 
é melhor que 90% dos "projetos de RAG" do me
rcado, que não têm avaliação.
- **Nível 
produção:** + M5–M7.
- **Sempre:** M8 (wr
iteup).
- **Esforço:** MVP ~2-3 fins de sema
na; completo ~6-8 semanas part-time (casa com
 o cronograma do mestrado).

## Kickoff (essa
 semana)
1. Cria o repo `rag-eval` + `docker-
compose` com Qdrant e Langfuse.
2. Baixa 20-3
0 resoluções da ANATEL (ou corpus escolhido
), ingesta.
3. Faz o M1 (baseline responde co
m citação) e **vê o primeiro trace no Lang
fuse**.
4. Começa a escrever 10 perguntas go
ld à mão — o resto do M2 vem depois.

> R
egra de ouro: **o gold set e o harness vêm c
edo.** Sem medição, é mais um projeto de R
AG bonito e indefensável — exatamente o qu
e você não quer.


