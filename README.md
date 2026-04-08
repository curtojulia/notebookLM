# Criação de um NotebookLM para guia de estudos de extração e preparação de dados
link para acessar o notebook: https://notebooklm.google.com/notebook/ae8da9d8-e5a6-4f7a-95c0-428a1ef6d4cb/preview
  O assunto do meu caderno foi desde a limpeza, normalização até a criação de chunks e suas características, decidi escolher esse tema em razão a um matéria que estou tendo na faculdade (Extração de preparação de dados), onde estamos aprendendo sobre Inteligência artificial, começando em como tratar valores nulos, duplicatas, outliers, até deixar uma base de dados perfeita para entregar para um LLM, onde serão feito chunks, de acordo com o documento, ou por largura fixa, ou por semântica,hibrído, por estrutura, e conhecendo melhor as boas práticas dessas técnicas, com o objetivo de me preparar para minha prova.

Fontes: Os PDFs das matérias disponibilizados pelo professor (2) e um arquivo de texto feito por mim, porém direcionado ao próprio notebook para ele saber como deve tratar os usuários e entender o contexto de criação desse caderno.

# Um exemplo de pergunta estratégica foi:
"O que é um corpus para assistente?" -  escolhida por mim

# resposta:
"Para a nossa prova, vocês precisam ter muito claro que um corpus para assistentes com LLMs não é simplesmente um amontoado de documentos ou um "dump" de texto bruto
. Ele é, na verdade, uma estrutura cuidadosamente organizada de unidades de conhecimento que já estão preparadas e prontas para serem consumidas e processadas pelo modelo de Inteligência Artificial
.
A diferença fundamental de um corpus para um texto comum está na intencionalidade: cada unidade de informação (os chunks) dentro dele deve ser autocontida o suficiente para gerar uma resposta útil, mas conectada o bastante para preservar o contexto maior do conteúdo
. Lembrem-se sempre de que a qualidade e a precisão de um assistente são determinadas muito antes de você escrever qualquer prompt ou configurar o modelo; tudo começa nessa preparação cuidadosa
.
Anotem aí as quatro características principais de um bom corpus que com certeza serão cobradas na avaliação:
Navegável pelo modelo: Possui uma estrutura que facilita enormemente a recuperação de informações relevantes pela máquina
.
Preserva o contexto: Consegue manter as relações semânticas (de significado) e as dependências entre os conceitos, mesmo com o texto fragmentado
.
Reduz a ambiguidade: Garante informações claras e evita interpretações conflitantes que fariam a IA "alucinar"
.
Orienta a resposta: Utiliza metadados e marcadores estruturais que guiam o modelo de linguagem exatamente na direção correta na hora de formular a resposta final
.
Ficou claro o conceito, pessoal? Entender o que é esse corpus é o primeiro passo para não errarmos na etapa de extração e preparação!" - Bem completa e de acordo com o que foi passado nos slides.

# Não tive dificuldade com as respostas, forem todas coerentes com a base de conhecimento passada

# Resumo estruturado para a minha prova:
1. Resumo Estruturado para a Prova
A. Criação de Corpus e Estratégias de Chunking
O que é um Corpus: Não é um simples amontoado de textos, mas uma estrutura organizada de unidades de conhecimento preparadas para a IA
. Um bom corpus deve ser navegável, preservar o contexto, reduzir a ambiguidade e orientar a resposta
.
A importância do Chunking: O chunking não é um recorte aleatório, mas uma decisão de modelagem que define o que fica junto e o que se perde
.
As 4 Estratégias de Chunking:
Largura Fixa: Divide o texto por um número fixo de caracteres/tokens. Usa overlap (sobreposição de 10-20%) para manter algum contexto, mas tem baixa coerência semântica
.
Estrutural: Usa a estrutura natural do documento (títulos, seções, listas)
. Recomenda-se impor um limite de 800-1200 tokens para evitar blocos gigantes
.
Semântica: Detecta mudanças de significado e tema para cortar o texto, ideal para conteúdos desorganizados ou transcrições de áudio
.
Híbrida: Combina todas as abordagens (inicia pela estrutura, impõe limite de tamanho e valida a semântica) e é o padrão da indústria
.
Metadados Inline: São marcações textuais (ex: [TEMA: Férias]) no início do chunk que orientam a IA, reduzem ambiguidades e melhoram a recuperação da informação
.
B. Mineração e Pré-processamento de Dados
Mineração de Dados (KDD): É o processo de extrair conhecimento útil de grandes volumes de dados
. Suas etapas são: Coleta e Limpeza, Transformação, Mineração e Avaliação
.
Tratamento de Dados Nulos e Duplicados: Valores nulos (ausentes) podem ser removidos ou imputados (usando médias/medianas) para não prejudicar o modelo
. Dados duplicados não acrescentam informação e enviesam o modelo, por isso são removidos usando funções como drop_duplicates() do Pandas
.
Tratamento de Outliers: São valores atípicos que fogem do padrão geral
. Podem ser identificados por:
Z-score: Mede a distância em desvios-padrão da média (geralmente cortes acima de 3)
.
Regra dos 3 Sigmas: Valores além de 3 desvios-padrão em distribuições normais
.
Boxplot (Dispersão): Usa os quartis e o IQR (Intervalo Interquartil) para demarcar limites visuais
.
Normalização de Dados: Coloca as variáveis na mesma escala para evitar que uma ofusque a outra na IA
.
Min-Max Scaling: Coloca os valores entre 0 e 1
.
Standard Scaling: Deixa os dados com média 0 e desvio padrão 1
.
Robust Scaling: Usa mediana e IQR, sendo a melhor técnica quando os dados possuem outliers
.

# Principais Conceitos
Corpus para LLM: Estrutura organizada de fragmentos de texto (chunks) preparados para consumo de um assistente de IA
.
Chunk: A unidade conceitual de informação que define o contexto do que o modelo vai ler
.
Overlap (Sobreposição): A técnica de repetir o final de um chunk no começo do próximo (geralmente 10-20%) para evitar perda de contexto
.
KDD (Knowledge Discovery in Databases): O processo completo de descobrir padrões válidos, novos e úteis a partir de dados
.
Metadados Inline: Etiquetas inseridas em texto simples dentro de cada arquivo para orientar as respostas do modelo
.
Valores Nulos (NaN): Lacunas na base de dados onde a informação está ausente ou é desconhecida
.
Imputação: Técnica de preencher um dado faltante (nulo) usando médias, medianas ou métodos matemáticos para não ter que deletar a linha inteira
.
Outlier: Ponto fora da curva; um valor muito atípico que pode ser fruto de um erro de medição ou de um evento raro, e que distorce análises se não for tratado
.
IQR (Intervalo Interquartil): Medida robusta de dispersão de dados, correspondente à distância entre o 1º quartil (25%) e o 3º quartil (75%)
. Usado no Robust Scaling e no Boxplot.


# Prompts reutilizáveis:
1- Como identificar e tratar outliers usando Z-score e Boxplot?
2- Qual técnica de normalização é a mais recomendada para a prova?
3- Pode explicar a diferença entre preencher e remover valores nulos?
4- O que é Normalização de Dados?
5- Qual a diferença entre média e mediana na imputação?
6- Como o Boxplot ajuda a visualizar outliers?
7- Como o assistente trata valores nulos e outliers?
8- Quais as vantagens de usar arquivos .txt ou .md?
9- Como implementar metadados inline para melhorar a recuperação?
10- Quais são os principais problemas que o chunking mal feito causa?
