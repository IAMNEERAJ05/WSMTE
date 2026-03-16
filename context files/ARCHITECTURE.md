# WSMTE — Architecture Specification

## Input
- Shape: [batch_size, 5, 9]
- 5 = timesteps (5-day sliding window)
- 9 = features [Close_d, Volume_d, RSI_d, MACD_d, BB_width_d, ROC_d,
                 Polarity_company, Polarity_market, Subjectivity]

---

## Full Architecture Diagram

```
Input [batch, 5, 9]
      │
      ├─────────────────────────────────────────────────────┐
      │                         │                           │
      ▼                         ▼                           ▼
LSTM Branch               TCN Branch                  GRU Branch
units=64                  filters=64                  units=64
activation=tanh           kernel_size=2               activation=tanh
recurrent_act=sigmoid     dilations=[1,2,4]           recurrent_act=sigmoid
dropout=0.2               padding=causal              dropout=0.2
recurrent_dropout=0.0     activation=relu             recurrent_dropout=0.0
return_sequences=False    dropout=0.2                 return_sequences=False
                          skip_connections=True
                          batch_norm=False
                          layer_norm=False
                          weight_norm=False
      │                         │                           │
      ▼ [batch, 64]             ▼ [batch, 64]              ▼ [batch, 64]
      │                         │                           │
      └─────────── MERGE ───────────────────────────────────┘
                     │
          Config A-G: Concatenate → [batch, 192]
          Config H:   PSO weighted sum → [batch, 64]
                      w1×LSTM + w2×TCN + w3×GRU (w1+w2+w3=1)
                     │
                     ▼
              Dense(64, relu)
              Dropout(0.2)
                     │ [batch, 64]
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   REGRESSION HEAD       CLASSIFICATION HEAD
   Dense(16, relu)        Dense(16, relu)
   Dense(1, linear)       Dense(1, sigmoid)
   Loss: MSE              Loss: BCE
          │                     │
          ▼                     ▼
   next-day Close         P(up) ∈ [0,1]
   (normalized)
```

---

## Combined Loss Function

```
L_total = (1 / 2σ₁²) × MSE + (1 / 2σ₂²) × BCE + log(σ₁) + log(σ₂)

Where:
  σ₁ = trainable noise parameter for regression task
  σ₂ = trainable noise parameter for classification task
  Both initialized to 1.0, learned during training
  Implementation: tf.Variable, trainable=True
```

Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics", CVPR 2018.

---

## TCN Receptive Field Calculation

```
kernel_size = 2
dilations = [1, 2, 4]
Receptive Field = 1 + (kernel_size - 1) × sum(dilations)
                = 1 + (2-1) × (1+2+4)
                = 1 + 7 = 8

Input length = 5 days
Receptive field = 8 > 5 ✓ (full coverage with margin)
```

---

## Ablation Study — 8 Configs

| Config | Features Used | Merge | Heads | Runs | Tests |
|--------|--------------|-------|-------|------|-------|
| A | Close_d, Volume_d, RSI_d, MACD_d, BB_width_d, ROC_d | concat | classification | 10 | Technicals-only floor baseline |
| B | A + polarity_company + subjectivity | concat | classification | 10 | Company sentiment contribution |
| C | All 9 features | concat | classification | 10 | Novelty 2 — market sentiment |
| D | All 9 features | concat | classification | 10 | Full features single-task confirmed |
| E | All 9 features | concat | classification only | 10 | Novelty 4 (no regression) |
| F | All 9 features | concat | regression only | 10 | Novelty 4 (no classification) |
| G | All 9 features | concat | both heads | 10 | Full WSMTE without PSO |
| H | All 9 features | PSO weighted | both heads | 30 | FINAL proposed model |

Config H = your final proposed model. Compare against Kotekar 0.5853.

---

## PSO Two-Stage Process (Config H only)

### Stage 1 — Initial Training
```
Train full model with simple concatenation (same as Config G)
Early stopping patience=15 on val_loss
Save best model weights
```

### Stage 2 — PSO Weight Search
```
Load saved model weights
Extract LSTM, TCN, GRU branch outputs on validation set
Freeze all model weights

PSO search:
  Search space: 3 dimensions [w1, w2, w3]
  Constraint: softmax(weights) to ensure sum=1
  Fitness: negative validation accuracy
  n_particles: 20
  iterations: 50
  options: c1=0.5, c2=0.3, w=0.9

Output: best [w1, w2, w3]
```

### Stage 3 — Fine-tuning
```
Unfreeze model
Set branch weights to PSO-found values
Fine-tune for up to 20 epochs
Early stopping patience=5 on val_loss
Evaluate on test set
```

---

## Model Parameter Count (approximate)

```
LSTM branch:      ~33,000 parameters
GRU branch:       ~25,000 parameters
TCN branch:       ~25,000 parameters
Shared Dense(64): ~12,500 parameters
Reg head:         ~1,100 parameters
Class head:       ~1,100 parameters
σ₁, σ₂:          2 parameters
─────────────────────────────────────
Total:            ~97,800 parameters
```

Small enough to train comfortably on 1090 samples without overfitting
given dropout=0.2 and early stopping.

---

## Output Shapes at Each Stage

```
Input:              [batch, 5, 9]
After LSTM:         [batch, 64]
After GRU:          [batch, 64]
After TCN:          [batch, 64]
After concat (A-G): [batch, 192]
After PSO (H):      [batch, 64]
After Dense(64):    [batch, 64]
After Dropout:      [batch, 64]
Regression output:  [batch, 1]
Classification out: [batch, 1]
```

---

## Activation Functions Summary

```
LSTM internal gates:      sigmoid (fixed, not configurable)
LSTM cell state:          tanh (fixed, not configurable)
GRU internal gates:       sigmoid (fixed, not configurable)
TCN conv layers:          relu
Shared Dense(64):         relu
Regression head Dense(16): relu
Regression output:        linear (no activation)
Classification Dense(16): relu
Classification output:    sigmoid
```

---

## Notes for Implementation

1. TCN implementation: use philipperemy/keras-tcn library
   Install: pip install keras-tcn
   Import: from tcn import TCN

2. LSTM and GRU: standard tf.keras.layers

3. For parallel branches, use Keras functional API not Sequential

4. σ₁ and σ₂ implementation:
   log_sigma1 = tf.Variable(0.0, trainable=True, name='log_sigma1')
   log_sigma2 = tf.Variable(0.0, trainable=True, name='log_sigma2')
   loss = (tf.exp(-log_sigma1)*mse + log_sigma1 +
           tf.exp(-log_sigma2)*bce + log_sigma2)

5. For Config A-G concatenation:
   merged = tf.keras.layers.Concatenate()([lstm_out, tcn_out, gru_out])

6. For Config H PSO merge (after PSO finds weights):
   merged = w1*lstm_out + w2*tcn_out + w3*gru_out
