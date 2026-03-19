# WSMTE — Architecture Specification

## Input
- Shape: [batch_size, 5, 11]
- 5 = timesteps (5-day sliding window)
- 11 = features [Close_d, Volume_d, RSI_d, MACD_d, BB_width_d, ROC_d,
                  Polarity_company, Polarity_company_max,
                  Polarity_market, Polarity_market_max, Subjectivity]

---

## Full Architecture Diagram

```
Input [batch, 5, 11]
      │
      ├─────────────────────────────────────────────────────┐
      │                         │                           │
      ▼                         ▼                           ▼
LSTM Branch               TCN Branch                  GRU Branch
units=32                  filters=32                  units=32
activation=tanh           kernel_size=2               activation=tanh
recurrent_act=sigmoid     dilations=[1,2,4]           recurrent_act=sigmoid
dropout=0.2               padding=causal              dropout=0.2
recurrent_dropout=0.0     activation=relu             recurrent_dropout=0.0
return_sequences=False    dropout=0.2                 return_sequences=False
                          skip_connections=True
                          batch_norm=False
                          layer_norm=False
                          weight_norm=True
      │                         │                           │
      ▼ [batch, 32]             ▼ [batch, 32]              ▼ [batch, 32]
      │                         │                           │
      └─────────── MERGE ───────────────────────────────────┘
                     │
          Config A-E: Concatenate → [batch, 96]
          Config F:   PSO weighted sum → [batch, 32]
                      w1×LSTM + w2×TCN + w3×GRU (w1+w2+w3=1)
                     │
                     ▼
              Dense(32, relu)
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
L_total = 0.3 × MSE + 0.7 × BCE

Where:
  MSE = mean squared error (regression head)
  BCE = binary cross-entropy (classification head)
  Weights are fixed — no trainable noise parameters
```

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

## Ablation Study — 6 Configs

| Config | Features Used | Merge | Heads | Runs | Purpose |
|--------|--------------|-------|-------|------|---------|
| A | Close_d, High_d, Low_d, Open_d, Volume_d | concat | classification | 30 | Denoised OHLCV price-only floor |
| B | Close_d, High_d, Low_d, Open_d, Volume_d + all sentiment | concat | classification | 30 | OHLCV + sentiment, no technicals |
| C | Close_d, Volume_d, RSI_d, MACD_d, BB_width_d, ROC_d | concat | classification | 30 | Technical indicators only |
| D | All 11 features | concat | classification | 30 | Full features, single-task |
| E | All 11 features | concat | both heads | 30 | Full WSMTE without PSO |
| F | All 11 features | PSO weighted | both heads | 30 | FINAL proposed model |

Config F = your final proposed model. Compare against Kotekar 0.5853.

---

## PSO Two-Stage Process (Config F only)

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
LSTM branch:      ~8,500 parameters
GRU branch:       ~6,500 parameters
TCN branch:       ~6,500 parameters
Shared Dense(32): ~3,100 parameters
Reg head:         ~530 parameters
Class head:       ~530 parameters
─────────────────────────────────────
Total:            ~25,700 parameters
```

Small enough to train comfortably on ~1090 samples without overfitting
given dropout=0.2 and early stopping.

---

## Output Shapes at Each Stage

```
Input:              [batch, 5, 11]
After LSTM:         [batch, 32]
After GRU:          [batch, 32]
After TCN:          [batch, 32]
After concat (A-E): [batch, 96]
After PSO (F):      [batch, 32]
After Dense(32):    [batch, 32]
After Dropout:      [batch, 32]
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

4. Fixed-weight MTL loss:
   loss = 0.3 * mse + 0.7 * bce
   No trainable σ parameters.

5. For Config A-E concatenation:
   merged = tf.keras.layers.Concatenate()([lstm_out, tcn_out, gru_out])

6. For Config F PSO merge (after PSO finds weights):
   merged = w1*lstm_out + w2*tcn_out + w3*gru_out
