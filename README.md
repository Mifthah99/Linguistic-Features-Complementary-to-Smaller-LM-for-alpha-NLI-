# BASELINE CODE:

Training:

To run training and create a new model: 	python train_anli.py 

To run training to update existing model:	python train_anli.py model_name.pkl

if you want to change number of epochs please use epochs as cmd line arg, ex :python train_anli.py model_name.pkl epochs=20


Evaluation:

If you don't want to train just evaluate:	python evaluate.py

if you want to evaluate specific model:		python evaluate model_name.pkl




#ADVANCED CODE:
Training: 
To run training and create new model:   python HYPO_LLM.py train

Evaluate:
If you don't want to train just evaluate: python HYPO_LLM.py



---

# Are Linguistic Features Complementary to Smaller Language Models for $\alpha$-NLI? An Empirical Study with Perceptron and T5
## Abstract
Abductive Natural Language Inference (α-NLI) is a challenging task requiring robust
commonsense reasoning to identify the mostplausible hypothesis explaining two given
observations. This paper investigates the effectiveness of traditional feature-based models versus smaller pre-trained language models (LLMs) on the α-NLI task. We first establish a baseline using a Perceptron model
and then augment it with engineered linguistic features, including Jaccard similarity, lexical overlap, and sentiment polarity, achieving approximately 52.22% development and 60% test accuracy. Subsequently, we train and evaluate T5-small and T5-base models under different prompting strategies. Our findings reveal that T5 models perform significantly better when presented with both hypotheses comparatively
in a single prompt (66.4% development, over 90% test accuracy), but struggle considerably when assessing individual hypothesis plausibility (around 37% for both sets). Crucially, a post-hoc analysis on T5’s misclassified examples found no clear correlation with the engineered linguistic features, suggesting that smaller LLMs rely on different, implicitly learned patterns for their reasoning, or
are more sensitive to query formulation than explicit surface-level cues. These results underscore the continued relevance of feature engineering for simpler models while highlighting the profound impact of prompt de-
sign on LLM performance in complex reasoning tasks.


## 1 Introduction
Abductive Natural Language Inference (α-NLI) is a pivotal task in natural language understanding, demanding the ability to infer the most plausible explanation from a set of observations
(Bhagavatula et al., 2020). α-NLI requires a more detailed understanding of cause and effects, making it a good benchmark to test inference capabilities.
Traditionally, NLI and other reasoning tasks in NLP have heavily relied on feature engineering.Large language models (LLMs), has revolutionized NLP by demonstrating remarkable capabilities in implicitly capturing complex linguistic and world knowledge through vast pretraining. This raises fundamental questions about the continued utility of hand-crafted features in an era dominated by powerful, general purpose LLMs.

Our research aims to address the following
key questions:
1. How effective is a feature-augmented Perceptron model in tackling the α-NLI task,
and what is the individual contribution
of various linguistic features to its performance?
2. How do smaller pre-trained LLMs (T5)
perform on α-NLI, and to what extent
does their performance depend on different
prompting strategies?
3. Is there a synergistic relationship between
explicit linguistic features and LLMs for α-NLI, or do LLMs render such features redundant? Specifically, can explicit features
shed light on the nature of LLM misclassifications?


We establish a baseline for α-NLI using a Perceptron model enriched with features such as
Jaccard coefficient, sentiment polarity, and lexical overlap. We provide a comparative analysis of T5 models under distinct prompting
paradigms, demonstrating their sensitivity to input formulation. Finally, through a post-hoc error analysis, we explore the relationship between traditional linguistic features and the errors made by smaller LLMs, revealing insights into their underlying reasoning mechanisms.

## 2 Experimental Setup

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
Perceptron classifier. This model takes a concatenation of several hand-crafted linguistic features as input and outputs a binary classification. The Perceptron was trained using a step
function and optimized using the Perceptron
learning rule. It was trained for 2000 epochs
with a learning rate of 0.4.
For each α-NLI instance, the features for
(O1, O2, H1) and (O1, O2, H2) were computed.
The model then effectively compares the feature vectors associated with H1 and H2 to make
a choice. We also tried to separately just generate predictions for single hypothesis using a
simple threshold based labeler which just says
if a hypothesis is conforms to a observation or
not.

2.3 T5 Models and Prompting
Strategies
We tried both the pre-trained T5-small (60
million parameters) and T5-base (220 million parameters) models from the Hugging Face
transformers library. Both models were finetuned for 10 epochs with a batch size of 512 using the AdamW optimizer and a learning rate
of 3e−5.

We explored two distinct prompting strategies:
1. Comparative Prompting (Strategy A): In this setup, both candidate hypotheses (H1 and H2) were presented simultaneously within a single input prompt, and
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
evaluated independently against the observations. The model was prompted twice
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
observations or hypotheses, could help with α-NLI task. However, initial experiments and
qualitative analysis indicated that the topic distributions generated by BERT TOPIC did not
effectively differentiate between plausible and
implausible hypotheses for the α-NLI task.
2.5 Post-Hoc Feature Analysis on LLM
Misclassifications
To gain insight into the nature of T5’s errors,
we performed a post-hoc analysis of the linguistic features for correctly and incorrectly classified examples. For every instance in the development and test sets, we computed the same
set of Jaccard similarity,negation, lexical overlap, and sentiment polarity features that were
used by the Perceptron model. Importantly,
these feature scores were not provided as input
to the T5 models during fine-tuning or inference. Instead, we analyzed the distributions of
these feature values across T5’s correct and incorrect predictions, aiming to identify if particular feature profiles correlated with misclassi-
fication. We looked for patterns or significant
differences in average feature scores that might
indicate an area of weakness for the LLMs that
traditional features could potentially illuminate.
3 Results and Discussion
3.1 Perceptron Performance
The Perceptron model, augmented with our engineered linguistic features, achieved an accuracy of 52.22% on the development set and
60.0% on the test set (initially only 51.2% for
dev set). These results demonstrate that a classical, feature-based approach can achieve a respectable baseline on the α-NLI task. The inclusion of additional features like Jaccard coefficient, sentiment polarity, and lexical overlap
was crucial for achieving these results, highlighting the Perceptron’s reliance on explicit linguistic cues for discerning plausibility in α-NLI. The
addition of sentence embeddings helped further
the dev score to 53.4%. This performance, underscores the inherent difficulty of the task and
the necessity of capturing complex relationships
even with explicit feature engineering.
Comparative Prompting (Strategy A)
When both candidate hypotheses were presented comparatively in a single prompt (Strategy A), T5 models achieved significantly higher
accuracy. T5-small reached 61.5% on the development set and 85.2% on the test set, while
T5-base achieved 66.4% on development and
over 90% on the test set (see Table ??). This
substantial improvement over the Perceptron
baseline highlights the T5 models’ remarkable
ability to leverage their pre-trained knowledge
and contextual understanding for abductive reasoning when provided with a direct comparative
choice. The high performance on the test set,
in particular, suggests that for α-NLI, explicitly
framing the task as a comparative selection is
highly effective for T5, allowing it to efficiently
identify the more plausible explanation among
candidates.
Individual Assessment Prompting
(Strategy B)
In stark contrast, when hypotheses were assessed individually (Strategy B), T5 models exhibited a drastic drop in performance, achieving accuracies around 37% for both development and test sets. This indicates that T5
models struggled considerably when forced to
evaluate the plausibility of a single hypothesis
in isolation without the immediate comparative
context. The poor performance suggests that
for α-NLI, the ability to directly contrast hypotheses within the prompt is critical for T5
to perform effective abductive reasoning, rather
than merely assessing the absolute plausibility
of individual statements. This implies that T5’s
strength in α-NLI lies more in its capacity for
relative comparison and ranking, which might
be implicitly learned during pre-training, rather
than absolute commonsense evaluation in a binary ”plausible/implausible” sense for a given
observation pair.
3.2 Post-Hoc Feature Influence on
LLM Errors
Despite our initial hypothesis that T5’s misclassifications might correlate with deficiencies
in understanding specific surface-level linguistic properties, our post-hoc analysis of feature
scores for misclassified examples did not reveal
a clear or consistent relationship. We examined
the distributions of Jaccard similarity, negation,
lexical overlap, and sentiment polarity for both
correctly and incorrectly predicted instances by
T5 (under Strategy A, where performance was
high), but found no discernible patterns that
would systematically explain the misclassifications based on these features.
This finding suggests that the factors leading
to T5’s errors on α-NLI may not be primarily
related to its inability to implicitly capture or
process these particular linguistic features. Instead, its performance appears to be more sen-
sitive to the formulation of the query itself, indicating that its strength lies in leveraging pretrained contextual understanding within a comparative framework, rather than relying on explicit feature patterns for individual hypothesis
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
Our comparative analysis reveals a clear distinction in how feature-based models and smaller
LLMs approach the α-NLI task. The Perceptron’s reliance on engineered features underscores the continued importance of explicit linguistic cues for simpler models. Conversely,
T5’s superior performance, especially under
comparative prompting, highlights the immense
power of transfer learning and contextual understanding inherent in LLMs. However, T5’s dramatic drop in accuracy when deprived of comparative context suggests that its reasoning capabilities are highly sensitive to how the task
is framed. Our post-hoc analysis further supports the notion that T5’s errors are not easily
explained by a lack of understanding of basic
linguistic features, implying that its reasoning
failures stem from deeper, more abstract semantic or commonsense gaps.
4 Conclusion
This study empirically investigated the performance of a feature-augmented Perceptron
model and smaller T5 language models on the
challenging Abductive Natural Language Inference (α-NLI) task. We demonstrated that a
Perceptron model, enriched with hand-crafted
features such as Jaccard similarity, lexical overlap, and sentiment polarity, can achieve a respectable baseline accuracy of 60% on the test
set, highlighting the continued relevance of feature engineering for traditional machine learning approaches.
Our experiments with T5-small and T5-
base models revealed a critical dependency on
prompt formulation. T5 models exhibited significantly higher performance (over 90% accuracy on the test set for T5-base) when presented with a comparative choice between two
hypotheses in a single prompt. However, their
accuracy plummeted to around 37% when asked
to evaluate individual hypothesis plausibility,
underscoring that for α-NLI, smaller LLMs are
highly adept at relative comparison but struggle with absolute assessments without explicit
comparative context. Our post-hoc analysis
further indicated that the errors made by T5
models were not easily explained by patterns
in traditional linguistic features, suggesting that
T5’s reasoning relies on more complex, implicitly learned representations rather than simple
surface-level cues that benefit Perceptrons.
The design of prompts and the way information is presented to LLMs are paramount,
potentially outweighing the benefit of explicit
feature injection or the analysis of surface-level
linguistic properties. Future work should explore more sophisticated methods of integrating explicit knowledge and features into LLMs
(e.g., via retrieval-augmented generation or specialized fine-tuning) to address their remaining challenges in complex reasoning, especially
when direct comparative contexts are unavailable. Further error analysis could delve into the
semantic and logical types of errors made by
LLMs to pinpoint specific reasoning shortcomings.
A Appendix
A.1 Detailed Feature Definitions
• Jaccard Similarity: For two strings SA
and SB , converted to sets of unique words
WA and WB after lowercasing and tok-
enization, the Jaccard similarity J(SA, SB )
is calculated as: J(SA, SB ) = |WA∩WB |
|WA∪WB | .
We computed six Jaccard similarity features: J(O1, H1), J(O2, H1), J(O1, H2),
J(O2, H2), J(O1, O2), and J(H1, H2).
• Lexical Overlap: This feature quantifies
the common words between different com-
ponents of the input. We calculated the
number of overlapping tokens (after lowercasing and punctuation removal) between
(O1, H1), (O2, H1), (O1, H2), (O2, H2),
and also between (O1, O2). These counts
were normalized by the length of the longer
component.
• Sentiment Polarity: We used list of positive and negative words to extract the sentiment polarity score (a real value, typically between -1 and 1 for negative to positive) for each O1, O2, H1, H2. Additional
features included the absolute difference in
sentiment between O1 and O2, and between
each observation and its respective hypothesis.
• Negation: This feature takes into consideration presence of words like not, no,
never, etc. which can cause the meaning of
the sentence to change. This was for each
O1, O2, H1, H2
• Embeddings: We also made used of sentence and word embeddings to see if the
helped with improving perceptron performance.
References
Chandra Bhagavatula, Ronan Le Bras, and Yejin
Choi. 2020. Abductive natural language inference. In Proceedings of the 58th Annual Meeting
of the Association for Computational Linguistics,
pages 3700–3712, Online. Association for Computational Linguistics.
