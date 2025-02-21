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
transformed data {
  vector[N] sigma_factor = inv(sqrt(to_vector(size)));
}
parameters {
  vector[N_lectin] a_lectin;
  vector[N_mouse] a_mouse;
  vector[N_vessel_type] a_vessel_type;
  vector[N_lectin_vessel_type] a_lectin_vessel_type;
  real<lower=0> sigma;
  real<lower=0> tau_vessel_type;
  real<lower=0> tau_lectin_vessel_type;
}
model {
  a_lectin ~ normal(0, 2);
  a_mouse ~ normal(0, 0.05);
  a_vessel_type ~ normal(0, tau_vessel_type);
  a_lectin_vessel_type ~ normal(0, tau_lectin_vessel_type);
  sigma ~ normal(0, 1);
  tau_vessel_type ~ normal(0, 0.2);
  tau_lectin_vessel_type ~ normal(0, 0.2);
  vector[N] yhat = a_lectin[lectin]
    + a_mouse[mouse]
    + a_vessel_type[vessel_type]
    + a_lectin_vessel_type[lectin_vessel_type];
  y ~ normal(yhat, sigma_factor * sigma);
}
generated quantities {
  array[N] real yrep;
  {
    vector[N] yhat = a_lectin[lectin]
      + a_mouse[mouse]
      + a_vessel_type[vessel_type]
      + a_lectin_vessel_type[lectin_vessel_type];
    yrep = normal_rng(yhat, sigma_factor * sigma);
  }
}
