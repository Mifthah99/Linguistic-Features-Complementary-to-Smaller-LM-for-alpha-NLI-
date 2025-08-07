BASELINE CODE:

Training:

To run training and create a new model: 	python train_anli.py 

To run training to update existing model:	python train_anli.py model_name.pkl

if you want to change number of epochs please use epochs as cmd line arg, ex :python train_anli.py model_name.pkl epochs=20


Evaluation:

If you don't want to train just evaluate:	python evaluate.py

if you want to evaluate specific model:		python evaluate model_name.pkl




ADVANCED CODE:
Training: 
To run training and create new model:   python HYPO_LLM.py train

Evaluate:
If you don't want to train just evaluate: python HYPO_LLM.py



---

# Are Linguistic Features Complementary to Smaller Language Models for $\alpha$-NLI? An Empirical Study with Perceptron and T5
Abstract
Abductive Natural Language Inference (α-
NLI) is a challenging task requiring robust
commonsense reasoning to identify the most
plausible hypothesis explaining two given
observations. This paper investigates the ef-
fectiveness of traditional feature-based mod-
els versus smaller pre-trained language mod-
els (LLMs) on the α-NLI task. We first es-
tablish a baseline using a Perceptron model
and then augment it with engineered lin-
guistic features, including Jaccard similar-
ity, lexical overlap, and sentiment polar-
ity, achieving approximately 52.22% devel-
opment and 60% test accuracy. Subse-
quently, we train and evaluate T5-small and
T5-base models under different prompting
strategies. Our findings reveal that T5 mod-
els perform significantly better when pre-
sented with both hypotheses comparatively
in a single prompt (66.4% development, over
90% test accuracy), but struggle consider-
ably when assessing individual hypothesis
plausibility (around 37% for both sets). Cru-
cially, a post-hoc analysis on T5’s misclassi-
fied examples found no clear correlation with
the engineered linguistic features, suggesting
that smaller LLMs rely on different, implic-
itly learned patterns for their reasoning, or
are more sensitive to query formulation than
explicit surface-level cues. These results un-
derscore the continued relevance of feature
engineering for simpler models while high-
lighting the profound impact of prompt de-
sign on LLM performance in complex rea-
soning tasks.
1 Introduction
Abductive Natural Language Inference (α-NLI)
is a pivotal task in natural language understand-
ing, demanding the ability to infer the most
plausible explanation from a set of observations
(Bhagavatula et al., 2020). α-NLI requires a
more detailed understanding of cause and ef-
fects, making it a good benchmark to test infer-
ence capabilities.
Traditionally, NLI and other reasoning tasks
in NLP have heavily relied on feature engineer-
ing.Large language models (LLMs), has revolu-
tionized NLP by demonstrating remarkable ca-
pabilities in implicitly capturing complex lin-
guistic and world knowledge through vast pre-
training. This raises fundamental questions
about the continued utility of hand-crafted fea-
tures in an era dominated by powerful, general-
purpose LLMs.
Our research aims to address the following
key questions:
1. How effective is a feature-augmented Per-
ceptron model in tackling the α-NLI task,
and what is the individual contribution
of various linguistic features to its perfor-
mance?
2. How do smaller pre-trained LLMs (T5)
perform on α-NLI, and to what extent
does their performance depend on different
prompting strategies?
3. Is there a synergistic relationship between
explicit linguistic features and LLMs for α-
NLI, or do LLMs render such features re-
dundant? Specifically, can explicit features
shed light on the nature of LLM misclassi-
fications?
We establish a baseline for α-NLI using a Per-
ceptron model enriched with features such as
Jaccard coefficient, sentiment polarity, and lex-
ical overlap. We provide a comparative anal-
ysis of T5 models under distinct prompting
paradigms, demonstrating their sensitivity to
input formulation. Finally, through a post-hoc
error analysis, we explore the relationship be-
tween traditional linguistic features and the er-
rors made by smaller LLMs, revealing insights
into their underlying reasoning mechanisms.
2 Experimental Setup
2.1 The α-NLI Task
The α-NLI task is formulated as a forced-choice
classification problem. Given two observations,
O1 and O2, and two candidate hypotheses, H1
and H2, the model must select the hypothesis
that provides the more plausible explanation for
the transition from O1 to O2. The dataset used
in our experiments is the official α-NLI dataset
(Bhagavatula et al., 2020), but we make use of
part of training data as test data as true labels
for the original test data are no longer available.
2.2 Basic Perceptron Model with
Engineered Features
Our baseline model is a basic (single-layer)
Perceptron classifier. This model takes a con-
catenation of several hand-crafted linguistic fea-
tures as input and outputs a binary classifica-
tion. The Perceptron was trained using a step
function and optimized using the Perceptron
learning rule. It was trained for 2000 epochs
with a learning rate of 0.4.
For each α-NLI instance, the features for
(O1, O2, H1) and (O1, O2, H2) were computed.
The model then effectively compares the fea-
ture vectors associated with H1 and H2 to make
a choice. We also tried to separately just gen-
erate predictions for single hypothesis using a
simple threshold based labeler which just says
if a hypothesis is conforms to a observation or
not.
2.3 T5 Models and Prompting
Strategies
We tried both the pre-trained T5-small (60
million parameters) and T5-base (220 mil-
lion parameters) models from the Hugging Face
transformers library. Both models were fine-
tuned for 10 epochs with a batch size of 512 us-
ing the AdamW optimizer and a learning rate
of 3e−5.
We explored two distinct prompting strate-
gies:
1. Comparative Prompting (Strategy
A): In this setup, both candidate hypothe-
ses (H1 and H2) were presented simulta-
neously within a single input prompt, and
the model was asked to directly choose the
better explanation.
Given these observations:
1: [O1]
2: [O2]
which hypothesis is more likely?
1: [H1]
2: [H2]
Answer with either ’1’ or ’2’
2. Individual Assessment Prompting
(Strategy B): Here, each hypothesis was
evaluated independently against the obser-
vations. The model was prompted twice
per α-NLI instance (once for H1, once for
H2).
Given these observations:
1: [O1]
2: [O2]
Does this hypothesis align
with observations?
hypothesis: [H1]
Answer with either ’0’ or ’1’
2.4 BERT TOPIC Exploration
We briefly explored the utility of BERT TOPIC
for extracting topic-based features to investigate
whether explicit topic information, derived from
observations or hypotheses, could help with α-
NLI task. However, initial experiments and
qualitative analysis indicated that the topic dis-
tributions generated by BERT TOPIC did not
effectively differentiate between plausible and
implausible hypotheses for the α-NLI task.
2.5 Post-Hoc Feature Analysis on LLM
Misclassifications
To gain insight into the nature of T5’s errors,
we performed a post-hoc analysis of the linguis-
tic features for correctly and incorrectly classi-
fied examples. For every instance in the devel-
opment and test sets, we computed the same
set of Jaccard similarity,negation, lexical over-
lap, and sentiment polarity features that were
used by the Perceptron model. Importantly,
these feature scores were not provided as input
to the T5 models during fine-tuning or infer-
ence. Instead, we analyzed the distributions of
these feature values across T5’s correct and in-
correct predictions, aiming to identify if partic-
ular feature profiles correlated with misclassi-
fication. We looked for patterns or significant
differences in average feature scores that might
indicate an area of weakness for the LLMs that
traditional features could potentially illuminate.
3 Results and Discussion
3.1 Perceptron Performance
The Perceptron model, augmented with our en-
gineered linguistic features, achieved an accu-
racy of 52.22% on the development set and
60.0% on the test set (initially only 51.2% for
dev set). These results demonstrate that a clas-
sical, feature-based approach can achieve a re-
spectable baseline on the α-NLI task. The in-
clusion of additional features like Jaccard co-
efficient, sentiment polarity, and lexical overlap
was crucial for achieving these results, highlight-
ing the Perceptron’s reliance on explicit linguis-
tic cues for discerning plausibility in α-NLI. The
addition of sentence embeddings helped further
the dev score to 53.4%. This performance, un-
derscores the inherent difficulty of the task and
the necessity of capturing complex relationships
even with explicit feature engineering.
Comparative Prompting (Strategy A)
When both candidate hypotheses were pre-
sented comparatively in a single prompt (Strat-
egy A), T5 models achieved significantly higher
accuracy. T5-small reached 61.5% on the de-
velopment set and 85.2% on the test set, while
T5-base achieved 66.4% on development and
over 90% on the test set (see Table ??). This
substantial improvement over the Perceptron
baseline highlights the T5 models’ remarkable
ability to leverage their pre-trained knowledge
and contextual understanding for abductive rea-
soning when provided with a direct comparative
choice. The high performance on the test set,
in particular, suggests that for α-NLI, explicitly
framing the task as a comparative selection is
highly effective for T5, allowing it to efficiently
identify the more plausible explanation among
candidates.
Individual Assessment Prompting
(Strategy B)
In stark contrast, when hypotheses were as-
sessed individually (Strategy B), T5 models ex-
hibited a drastic drop in performance, achiev-
ing accuracies around 37% for both develop-
ment and test sets. This indicates that T5
models struggled considerably when forced to
evaluate the plausibility of a single hypothesis
in isolation without the immediate comparative
context. The poor performance suggests that
for α-NLI, the ability to directly contrast hy-
potheses within the prompt is critical for T5
to perform effective abductive reasoning, rather
than merely assessing the absolute plausibility
of individual statements. This implies that T5’s
strength in α-NLI lies more in its capacity for
relative comparison and ranking, which might
be implicitly learned during pre-training, rather
than absolute commonsense evaluation in a bi-
nary ”plausible/implausible” sense for a given
observation pair.
3.2 Post-Hoc Feature Influence on
LLM Errors
Despite our initial hypothesis that T5’s mis-
classifications might correlate with deficiencies
in understanding specific surface-level linguis-
tic properties, our post-hoc analysis of feature
scores for misclassified examples did not reveal
a clear or consistent relationship. We examined
the distributions of Jaccard similarity, negation,
lexical overlap, and sentiment polarity for both
correctly and incorrectly predicted instances by
T5 (under Strategy A, where performance was
high), but found no discernible patterns that
would systematically explain the misclassifica-
tions based on these features.
This finding suggests that the factors leading
to T5’s errors on α-NLI may not be primarily
related to its inability to implicitly capture or
process these particular linguistic features. In-
stead, its performance appears to be more sen-
sitive to the formulation of the query itself, in-
dicating that its strength lies in leveraging pre-
trained contextual understanding within a com-
parative framework, rather than relying on ex-
plicit feature patterns for individual hypothesis
assessment. It is plausible that smaller LLMs
like T5 already implicitly encode much of the
information conveyed by these simple features.
Therefore, their classification errors might stem
from more complex reasoning challenges, such
as subtle logical inconsistencies, less frequent
commonsense patterns not extensively covered
in pre-training, or limitations in handling very
specific causal structures within the α-NLI task
that go beyond simple feature-level analysis.
3.3 Comparative Analysis and Insights
Our comparative analysis reveals a clear distinc-
tion in how feature-based models and smaller
LLMs approach the α-NLI task. The Percep-
tron’s reliance on engineered features under-
scores the continued importance of explicit lin-
guistic cues for simpler models. Conversely,
T5’s superior performance, especially under
comparative prompting, highlights the immense
power of transfer learning and contextual under-
standing inherent in LLMs. However, T5’s dra-
matic drop in accuracy when deprived of com-
parative context suggests that its reasoning ca-
pabilities are highly sensitive to how the task
is framed. Our post-hoc analysis further sup-
ports the notion that T5’s errors are not easily
explained by a lack of understanding of basic
linguistic features, implying that its reasoning
failures stem from deeper, more abstract seman-
tic or commonsense gaps.
4 Conclusion
This study empirically investigated the per-
formance of a feature-augmented Perceptron
model and smaller T5 language models on the
challenging Abductive Natural Language Infer-
ence (α-NLI) task. We demonstrated that a
Perceptron model, enriched with hand-crafted
features such as Jaccard similarity, lexical over-
lap, and sentiment polarity, can achieve a re-
spectable baseline accuracy of 60% on the test
set, highlighting the continued relevance of fea-
ture engineering for traditional machine learn-
ing approaches.
Our experiments with T5-small and T5-
base models revealed a critical dependency on
prompt formulation. T5 models exhibited sig-
nificantly higher performance (over 90% accu-
racy on the test set for T5-base) when pre-
sented with a comparative choice between two
hypotheses in a single prompt. However, their
accuracy plummeted to around 37% when asked
to evaluate individual hypothesis plausibility,
underscoring that for α-NLI, smaller LLMs are
highly adept at relative comparison but strug-
gle with absolute assessments without explicit
comparative context. Our post-hoc analysis
further indicated that the errors made by T5
models were not easily explained by patterns
in traditional linguistic features, suggesting that
T5’s reasoning relies on more complex, implic-
itly learned representations rather than simple
surface-level cues that benefit Perceptrons.
The design of prompts and the way infor-
mation is presented to LLMs are paramount,
potentially outweighing the benefit of explicit
feature injection or the analysis of surface-level
linguistic properties. Future work should ex-
plore more sophisticated methods of integrat-
ing explicit knowledge and features into LLMs
(e.g., via retrieval-augmented generation or spe-
cialized fine-tuning) to address their remain-
ing challenges in complex reasoning, especially
when direct comparative contexts are unavail-
able. Further error analysis could delve into the
semantic and logical types of errors made by
LLMs to pinpoint specific reasoning shortcom-
ings.
A Appendix
A.1 Detailed Feature Definitions
• Jaccard Similarity: For two strings SA
and SB , converted to sets of unique words
WA and WB after lowercasing and tok-
enization, the Jaccard similarity J(SA, SB )
is calculated as: J(SA, SB ) = |WA∩WB |
|WA∪WB | .
We computed six Jaccard similarity fea-
tures: J(O1, H1), J(O2, H1), J(O1, H2),
J(O2, H2), J(O1, O2), and J(H1, H2).
• Lexical Overlap: This feature quantifies
the common words between different com-
ponents of the input. We calculated the
number of overlapping tokens (after lower-
casing and punctuation removal) between
(O1, H1), (O2, H1), (O1, H2), (O2, H2),
and also between (O1, O2). These counts
were normalized by the length of the longer
component.
• Sentiment Polarity: We used list of pos-
itive and negative words to extract the sen-
timent polarity score (a real value, typi-
cally between -1 and 1 for negative to pos-
itive) for each O1, O2, H1, H2. Additional
features included the absolute difference in
sentiment between O1 and O2, and between
each observation and its respective hypoth-
esis.
• Negation: This feature takes into con-
sideration presence of words like not, no,
never, etc. which can cause the meaning of
the sentence to change. This was for each
O1, O2, H1, H2
• Embeddings: We also made used of sen-
tence and word embeddings to see if the
helped with improving perceptron perfor-
mance.
References
Chandra Bhagavatula, Ronan Le Bras, and Yejin
Choi. 2020. Abductive natural language infer-
ence. In Proceedings of the 58th Annual Meeting
of the Association for Computational Linguistics,
pages 3700–3712, Online. Association for Compu-
tational Linguistics.
