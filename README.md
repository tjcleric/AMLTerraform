# AMLTerraform
## The Impact of Realistic Laundering Subgraph Perturbations on Graph Neural Network Based Anti-Money Laundering Systems
As financial institutions adopt more sophisticated Anti-Money Laundering (AML) techniques, such as the deployment of Graph Neural Networks (GNNs) to detect patterns, laundering behavior is likely to evolve. In this paper, we present a novel perturbation framework that models laundering as an evasion-based, restricted black-box process. Our tool systematically alters labeled laundering subgraphs through a set of parameterized graph actions (intermediary injection, merging, and splitting) designed to simulate realistic laundering adaptations. We apply our framework to one of the AMLWorld synthetic transaction datasets to generate multiple perturbed versions defined by a set of parameterized preset configuration files. We then evaluate the impact of these perturbations on two MEGA-GNN variants of the current state-of-the-art in temporal multigraph-compatible GNN architectures. Our results show that realistic structural perturbations can impact performance and serve as a valuable tool to evaluate model adaptability and robustness. Our work aims to contribute to a deeper understanding of the evolutionary dynamics between AML systems and laundering behavior.

## Usage
Perturbation tool: terraform.py

## References
AMLWorld datasets: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
```
@misc{altman2024realisticsyntheticfinancialtransactions,
      title={Realistic Synthetic Financial Transactions for Anti-Money Laundering Models}, 
      author={Erik Altman and Jovan Blanuša and Luc von Niederhäusern and Béni Egressy and Andreea Anghel and Kubilay Atasu},
      year={2024},
      eprint={2306.16424},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2306.16424}, 
}
```


MEGA-GNN Repository: https://github.com/hcagri/MEGA-GNN
```
@misc{bilgi2024multigraphmessagepassingbidirectional,
      title={Multigraph Message Passing with Bi-Directional Multi-Edge Aggregations}, 
      author={H. Çağrı Bilgi and Lydia Y. Chen and Kubilay Atasu},
      year={2024},
      eprint={2412.00241},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.00241}, 
}
```

## TODO
- Usage

