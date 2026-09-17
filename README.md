##Statement
We used AI tools to assist with grammar correction, terminology suggestions, code debugging, and a small portion of repetitive coding tasks (e.g., script formatting).  All model design, core code logic, analysis, and final content were solely determined by the authors.


## 🔧 Key Dependencies

Python 3.10.16

Below are the key package versions used in this project:

- datasets==3.5.0
- numpy==2.1.2
- pandas==2.2.3
- tensorflow==2.19.0
- torch==2.5.1+cu121
- transformers==4.51.3

Make sure to install the above packages to ensure compatibility and reproducibility.

## 📁 File Overview

- **`Prediction_Case.log`**  
  Contains model inference results with detailed confidence scores for each prediction.

- **`Train_Dymsampling.py`**  
  Implements the training procedure using the **Dynamic Sampling** strategy.

- **`Train_linearsampling.py`**  
  Implements training using a **Linear Curriculum Sampling** strategy.

- **`Test_CSmulti.py`**  
  Evaluation script using the **CS_multi** remasking strategy.  
  _(Refer to Appendix D in the paper for more details.)_

- **`Test_CSsingle.py`**  
  Evaluation script using the **CS_single** remasking strategy.  
  _(Refer to Appendix D in the paper for more details.)_
