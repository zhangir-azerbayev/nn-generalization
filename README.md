# Neural Network Generalization Reading List 
Why do neural networks generalize? 

Consider the problem of learning to map some input $X$ to an output $Y$ given some finite number of training examples $M  = \{(x,y)\}$. All learning algorithms are equally bad at this problem, because in general, knowing $M$ tells you nothing about what the rest of $Y^X$ looks like (where $Y^X$ denotes the set of functions from $X$ into $Y$). This is the famous no-free lunch theorem in statistical learning theory. David Hume expressed some version of this theorem before we had the tools to give it a rigorous basis. 

Does this mean learning from data is impossible? Clearly not, because humans do it all the time. The key insight is that we don't care about abritrary learning problems, rather we care about structured learning problems, i.e problems where $Y^X$ is in some sense compressible and thus a finite number of $(x,y)$ pairs do in fact tell you something about the rest of $Y^X$. 

Given $M$, the ideal learner would find the shortest program that generates the dataset, where "short" can be formalized as Kolmogorov complexity. This requirement essentially encodes Occam's razor (see also [Solomonoff induction](https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference). However, a basic result in algorithmic information theory states that finding such a shortest program is uncomputable. 

Does this mean learning from data is uncomputable? Not quite, because it is not necessary that our learner search the entirety of program space. Just as we can get away with using finite-precision floats while the reals are uncomputable, is there some dense subset of program space we can efficiently search through? 

The last 10 years of AI research have empirically that searching through the space of neural networks by using stochastic gradient descent to optimize a loss function over the data is a fantastic practical approximation to computing the shortest data-generating program. This fact is remarkable, and is certainly the most important discovery in the history of AI. Future generations may even judge it as the most important discovery in human history. 

It is important to think of neural networks not only as a special class of functions, such as logistic regressors or support vector machines, but as a programming paradigm. That is, a very clever programmer could manually deduce the weights that yield an arbitrary program, just as he might write lines of C or Python. Training a neural network is truly akin to searching program space. (Nb. that a single feedforward neural network is a bit special compared to a C program, because defining a feedforward architecture *precisely specifies the number of instructions executing your program consists of, whereas even a very short C program can execute arbitrarily many instructions. This may seem like a great limitation, however it is not for two reasons 1) you can always make your neural network bigger if you need to do a bigger computation and 2) if you really want to execute an unbounded number of instructions, there are many promising approaches including recurrence and language model cascades.)

It is easy to see that the space of neural networks provides good coverage of program space, but it is a mystery why optimizing a loss function with SGD is a useful search procedure that effectively approximates looking for the shortest data-generating program. The readings in this list comprise some of our early efforts to unravel this mystery. 


## Preliminaries and Overviews 
| Author | Title | Year | Type | Remarks |
|--------|-------|------|------|---------|
|Hardt and Recht| ["Generalization"](https://mlstory.org/generalization.html), in *Patterns, Prediction, and Action*| 2021|book chapter|Classical ideas in statistical learning theory and empirical phenomena in deep learning |
|Friedman, Hastie, and Tibshirani | "Model Assesment and Selection", in *Elements of Statistical Learning* | 2008 |book chapter| A good overview of classical learning theory. Classical learning theory explains traditional ML algorithms, but seems to contradict empirical phenomena in deep learning| 
|Poggio| [Theoretical issues in deep networks](https://www.pnas.org/doi/10.1073/pnas.1907369117) | 2020 | paper | By one of the founders of computational neuorscience. |
|Roberts, Yaida, and Hanin| [Principles of Deep Learning Theory](https://arxiv.org/abs/2106.10165) | 2022 | book | A tour de force. One of the few compelling analyses of the dynamics of deep networks.|

## Scaling Laws and Pre-training
| Author | Title | Year | Type | Remarks |
|--------|-------|------|------|---------|
|Kaplan et al.| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)|2020| paper | | 
|Tripuraneni et al.| [On the Theory of Transfer Learning: The Importance of Task Diversity](https://arxiv.org/abs/2006.11650)|2020| paper | |
|Bommasani et al.| [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) | 2021 | monograph | |
|Wei et al. | [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) | 2022 | paper| |

## Optimization Dynamics 
| Author | Title | Year | Type | Remarks |
|--------|-------|------|------|---------|
|Zhang et al.| [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) | 2016 | paper | |
|Zhang et al.| [Theory of Deep Learning III: Generalization Properties of SGD](https://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-067.pdf) | 2017 | monograph | | 
|Jacot et al.| [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) | 2018 | paper | | 
|Wang and Isola| [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) | 2020 | paper | | 
|Power et al.| [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) | 2021 | paper | | 


## Deep Double Descent 
| Author | Title | Year | Type | Remarks |
|--------|-------|------|------|---------|
|Hubinger|[Understanding "Deep Double Descent"](https://www.lesswrong.com/posts/FRv7ryoqtvSuqBxuT/understanding-deep-double-descent) |2019|blog post| |
|Belkin et al.| [Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)|2019 | paper | | 
|Nakkiran et al.| [Deep Double Descent: Where Bigger Models and More Data Hurt](https://mltheory.org/deep.pdf) | 2019 | paper | The original deep double descent paper| 
|Keskar et al.|[On Large-Batch Training For Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) | 2017 | paper | | 
|Dinh et al.|[Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933)| 2017 | paper | | 
| Neyshabur et al.| [Exploring Generalization in Deep Learning](https://papers.nips.cc/paper/2017/hash/10ce03a1ed01077e3e289f3e53c72813-Abstract.html) | 2017 | paper | Scale normalization, connection between sharpness and PAC-Bayes theory. | 
|Frankle and Carbin | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | 2018 | paper | |

## Data Manifolds 
| Author | Title | Year | Type | Remarks |
|--------|-------|------|------|---------|
|Carlsson et al. | [On the Local Behavior of Spaces of Natural Images](http://math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/mumford-carlsson%20et%20al.pdf) | 2008 | paper | |
|Olah | [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | 2014 | blog post| 
|Fefferman et al.| [Testing the manifold hypothesis](https://www.ams.org/journals/jams/2016-29-04/S0894-0347-2016-00852-4/) | 2016 | paper | |

## Generalization Bounds
| Author | Title | Year | Type | Remarks |
|--------|-------|------|------|---------|
|Dziugaite and Roy| [Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/abs/1703.11008) | 2017 | paper | | 
|Arora et al. | [Stronger generalization bounds for deep nets via a compression approach](https://arxiv.org/abs/1802.05296) | 2018 | paper | | 
|Bubeck and Sellke| [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806) | 2021 | paper | Overparametrized models learn smooth functions. Distinguished among theory papers in that the result both seems quite general and quite relevant.|
|HaoChen et al.|[Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss](https://arxiv.org/abs/2106.04156) | 2021 | paper |  | 
