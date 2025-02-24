data {
  int N; 
  int N_lectin; 
  int N_mouse; 
  int N_vessel_type; 
  int N_lectin_vessel_type; 
  array[N] int size;
  vector[N] y;
  array[N] int lectin;
  array[N] int mouse;
  array[N] int vessel_type;
  array[N] int lectin_vessel_type;
}
parameters {
  vector[N_vessel_type-1] a_vessel_type_free;
  vector[N_lectin_vessel_type-2] a_lectin_vessel_type_free;
  real k;
  real<lower=0> sigma;
  real<lower=0> tau_vessel_type;
  real<lower=0> tau_lectin_vessel_type;
}
transformed parameters {
  vector[N_vessel_type] a_vessel_type;
  vector[N_lectin_vessel_type] a_lectin_vessel_type;
  a_vessel_type[1] = 0;
  a_vessel_type[2:N_vessel_type] = a_vessel_type_free;
  a_lectin_vessel_type[1:2] = rep_vector(0, 2);
  a_lectin_vessel_type[3:N_lectin_vessel_type] = a_lectin_vessel_type_free;
}
model {
  k ~ normal(0, 0.5);
  a_vessel_type_free ~ normal(0, tau_vessel_type);
  a_lectin_vessel_type_free ~ normal(0, tau_lectin_vessel_type);
  sigma ~ normal(0, 1);
  tau_vessel_type ~ normal(0, 0.5);
  tau_lectin_vessel_type ~ normal(0, 0.5);
  vector[N] yhat = k
    + a_vessel_type[vessel_type]
    + a_lectin_vessel_type[lectin_vessel_type];
  y ~ normal(yhat, sigma);
}
generated quantities {
  array[N] real yrep;
  {
    vector[N] yhat = k
      + a_vessel_type[vessel_type]
      + a_lectin_vessel_type[lectin_vessel_type];
    yrep = normal_rng(yhat, sigma);
  }
}
