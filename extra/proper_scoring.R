library(reticulate)
library(scoringRules)
#library(scoringutils)

pickle = import("pickle")

path = "/Users/Guille/Desktop/caiso_power/results/predictions/"
name = "567-63_1111_00_11_12_pred_v3.pkl"


data_   = py_load_object(paste(path, name, sep = ''), pickle = "pickle")
Y_      = data_[[1]]
M_hat_  = data_[[2]]
S2_hat_ = data_[[3]]
Y_hat_  = data_[[4]]
Y_joint_hat_  = data_[[5]]

dim(Y_)
dim(M_hat_)
dim(S2_hat_)
dim(Y_hat_)

N_obvs    = dim(Y_hat_)[1]
N_zones   = dim(Y_hat_)[2]
N_hours   = dim(Y_hat_)[3]
N_samples = dim(Y_hat_)[4]

quantiles_ = c(0.05, 0.25, 0.5, 0.75, 0.95)

CRPS_ = array(rep(1, N_zones*N_obvs*N_hours), dim = c(N_zones, N_hours))
LS_   = array(rep(1, N_zones*N_obvs*N_hours), dim = c(N_zones, N_hours))

for (i_zone in 1:N_zones){
  for (i_hour in 1:N_hours){
    y_     = as.vector(Y_[,i_zone,i_hour])
    y_hat_ = as.matrix(Y_hat_[,i_zone,i_hour,])
    y_joint_hat_ = as.matrix(Y_joint_hat_[,i_zone,i_hour,])
    
    CRPS_[i_zone, i_hour] = mean(crps_sample(y_, y_joint_hat_))
    LS_[i_zone, i_hour]   = mean(logs_sample(y_, y_joint_hat_))

  }
}
print(mean(CRPS_))
print(colMeans(LS_))

ES_ = array(rep(1, N_obvs*N_hours), 
              dim = c(N_obvs, N_hours))
VS_ = array(rep(1, N_obvs*N_hours), 
            dim = c(N_obvs, N_hours))

for (i_obv in 1:N_obvs){
  for (i_hour in 1:N_hours){
    y_     = Y_[i_obv,,i_hour]
    y_hat_ = Y_hat_[i_obv,,i_hour,]
    y_joint_hat_ = Y_joint_hat_[i_obv,,i_hour,]
    ES_[i_obv, i_hour] = es_sample(y_, y_joint_hat_)
    VS_[i_obv, i_hour] = vs_sample(y_, y_joint_hat_, p = .5)
  }
}

print(mean(colMeans(ES_)))
print(mean(colMeans(VS_)))


# 567-63_1111_00_11_13_pred.pkl    - 342.71, 220.88, 155.67
# 567-63_1110_00_11_13_pred_v2.pkl - 342.60, 207.08, 155.15
# 567-63_1110_00_11_13_pred.pkl    - 350.45, 221.83, 158.35

# 567-63_1111_00_11_12_pred_v2.pkl - 303.73, 138.34, 138.64
# 567-63_1111_00_11_12_pred.pkl    - 298.98, 153.78, 135.98
# 567-63_1110_00_11_12_pred_v2.pkl - 306.59, 143.41, 139.15
# 567-63_1110_00_11_12_pred.pkl    - 301.78, 161.04, 136.54

# 567-63_1111_00_11_10_pred_v2.pkl - 309.94, 143.10, 141.46
# 567-63_1111_00_11_10_pred.pkl    - 302.15, 139.86, 137.81
# 567-63_1110_00_11_10_pred_v2.pkl - 309.33, 146.73, 140.61
# 567-63_1110_00_11_10_pred.pkl    - 303.30, 144.74, 137.65

print(mean(colMeans(colMeans(CRPS_))))
print(mean(colMeans(colMeans(LS_))))
