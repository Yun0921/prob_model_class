/*
 * Author: Paul Horton
 * Copyright: Paul Horton 2021, All rights reserved.
 * Created: 20211201
 * Updated: 20240603
 * Licence: GPLv3
 * Description: Simple demonstration of a Bayesian way to guess at the number of components
 * behind a sample of numerical data.
 * Compile:  gcc -Wall -O3 -o Gaussian_poolOrNot Gaussian_poolOrNot.c GSLfun.c -lgsl -lgslcblas -lm
 * Environment: $GSL_RNG_SEED
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "GSLfun.h"

/* ───────────  Global definitions and variables  ────────── */
#define DATA_N 50
#define CDF_GAUSS_N 20
#define CDF_GAMMA_N 10
#define CDF_JBETA_N 40

typedef struct {
  double mixCof;     // 代表兩個分布中第一個高斯分布的權重
  Gauss_params Gauss1; // 第一個高斯分布的參數
  Gauss_params Gauss2; // 第二個高斯分布的參數
} Gauss_mixture_params;

const Gauss_params mu_prior_params = {0.0, 4.0}; // (平均，標準差)
const double sigma_prior_param_a = 0.5;         // gamma分布的形狀參數
const double sigma_prior_param_b = 2.0;         // gamma分布的尺度參數

double data[DATA_N];
const uint dataN = DATA_N;

enum modelNames { POOLED, DIFFER };

const uint sampleRepeatNum = 2000000;

/* ───────────  Functions to help summarize or dump the data  ────────── */

double data_sample_mean() {
  double mean = 0.0;
  for (uint i = 0; i < dataN; ++i) {
    mean += data[i];
  }
  return mean / (double)dataN;
}

double data_sample_variance() {
  double mean = data_sample_mean();
  double var = 0.0;
  for (uint i = 0; i < dataN; ++i) {
    double diff = data[i] - mean;
    var += diff * diff;
  }
  return var / (double)(dataN);
}

// ternary CMP function for use with qsort
int CMPdata(const void *arg1, const void *arg2) {
  return (*(double *)arg1 < *(double *)arg2)   ? -1
         : (*(double *)arg2 < *(double *)arg1) ? +1
                                             : /* else    *arg1 == *arg2   */ 0;
}

void data_print() {
  qsort(data, dataN, sizeof(double), CMPdata);
  for (uint i = 0; i < dataN; ++i) {
    printf("%u\t+%5.3f ", i, data[i]);
  }
}

/* ───────────  Functions used for sampling/generating data   ────────── */

Gauss_params prior_Gauss_params_sample() { // 用來產生一組可能的高斯分布參數
  Gauss_params params;
  params.mu = GSLfun_ran_gaussian(mu_prior_params);
  params.sigma =
      sigma_of_precision(GSLfun_ran_gamma(sigma_prior_param_a, sigma_prior_param_b));
  return params;
}

Gauss_mixture_params prior_Gauss_mixture_params_sample() {
  Gauss_mixture_params params;
  params.mixCof = GSLfun_ran_beta_Jeffreys();
  params.Gauss1 = prior_Gauss_params_sample();
  params.Gauss2 = prior_Gauss_params_sample();
  return params;
}

void data_generate_1component(Gauss_params params) { // 產生一組可能的高斯分布參數
  for (uint i = 0; i < dataN; ++i) {
    data[i] = GSLfun_ran_gaussian(params);
  }
}

void data_generate_2component(Gauss_mixture_params params) {
  for (uint i = 0; i < dataN; ++i) {
    data[i] = GSLfun_ran_gaussian(gsl_ran_flat01() < params.mixCof ? params.Gauss1
                                                                 : params.Gauss2); // 用gsl_ran_flat01產生均勻隨機數來決定要用哪個分布
  }
}

/* ───────────  Numerical integration precomputation  ────────── */

// Arrays to hold precomputed values.
double cdfInv_Gauss[CDF_GAUSS_N];
const double cdf_Gauss_n = CDF_GAUSS_N;
double cdfInv_gamma[CDF_GAMMA_N];
const double cdf_gamma_n = CDF_GAMMA_N;
double cdfInv_JBeta[CDF_JBETA_N];
const double cdf_JBeta_n = CDF_JBETA_N;

//  Precompute the cumulative probabilities of μ and σ discrete values.
//  The probabilities depend on the current prior_params values
void cdfInv_precompute() {
  double x;
  // Since Normal range is unbounded, precompute cdfInv for vals:  ¹⁄₍ₙ₊₁₎...ⁿ⁄₍ₙ₊₁₎
  for (uint i = 0; i < cdf_Gauss_n; ++i) {
    x = (i + 1) / (double)(1 + cdf_Gauss_n);
    cdfInv_Gauss[i] = gsl_cdf_gaussian_Pinv(x, mu_prior_params.sigma);
  }
  for (uint i = 0; i < cdf_gamma_n; ++i) {
    x = i / (double)(cdf_gamma_n);
    cdfInv_gamma[i] =
        gsl_cdf_gamma_Pinv(x, sigma_prior_param_a, sigma_prior_param_b);
    // printf( "cdfInv_Gamma[%u]= %g\n", i, cdfInv_gamma[i] );
  }
  for (uint i = 0; i < cdf_JBeta_n; ++i) {
    // By symmetry, only need Beta values for p ≦ 0.5.  For example p=0.8, is the same p=0.2 with Gauss components swapped.
    x = 0.5 * i / (double)(cdf_JBeta_n);
    cdfInv_JBeta[i] = gsl_cdf_beta_Pinv(x, 0.5, 0.5);
    // printf( "cdfInv_JBeta[%u]= %g\n", i, cdfInv_JBeta[i] );
  }
}

/* ───────────  Probability Computations on the Data  ────────── */

// Return Ｐ[D|μ,σ]
double prob_data_given_1Gauss(const Gauss_params params) { // 用來算聯合機率likelihood
  double prob = 1.0;
  for (uint d = 0; d < dataN; ++d) {
    prob *= GSLfun_ran_gaussian_pdf(data[d], params);
  }
  return prob;
}

// Return Ｐ[D|m,μ₁,σ₁,μ₂,σ₂]
double prob_data_given_2Gauss(const double mixCof, const Gauss_params Gauss1,
                              const Gauss_params Gauss2) {
  double prob = 1.0;
  for (uint i = 0; i < dataN; ++i) {
    prob *= (1 - mixCof) * GSLfun_ran_gaussian_pdf(data[i], Gauss2) +
             mixCof * GSLfun_ran_gaussian_pdf(data[i], Gauss1);
  }
  return prob;
}

// Return maximum likelihood of the data using a single Gaussian
double data_Gauss1_maxLikelihood() {
  Gauss_params params = {data_sample_mean(), data_sample_variance()};
  return prob_data_given_1Gauss(params);
}

// Function to calculate the log likelihood
double log_likelihood_2Gauss(const double *params, const double *data, int dataN) {
   double logL = 0.0;
   double mixCof = params[0];
   Gauss_params Gauss1 = {params[1], params[2]};
   Gauss_params Gauss2 = {params[3], params[4]};
 
   for (int i = 0; i < dataN; ++i) {
     double prob = (1 - mixCof) * GSLfun_ran_gaussian_pdf(data[i], Gauss2) +
                   mixCof * GSLfun_ran_gaussian_pdf(data[i], Gauss1);
     logL += log(prob > 1e-100 ? prob : 1e-100);
   }
   return logL;
}

// Function to perform one Newton-Raphson update step
void update_params(double *params, const double *data, int dataN) {
   double mixCof = params[0];
   double mu1 = params[1];
   double sigma1 = params[2];
   double mu2 = params[3];
   double sigma2 = params[4];
 
   double dL_d_mixCof = 0.0;
   double dL_d_mu1 = 0.0;
   double dL_d_mu2 = 0.0;
 
   for (int n = 0; n < dataN; ++n) {
     Gauss_params Gauss1 = {mu1, sigma1};
       Gauss_params Gauss2 = {mu2, sigma2};
     double p1_n = mixCof * GSLfun_ran_gaussian_pdf(data[n], Gauss1);
     double p2_n = (1 - mixCof) * GSLfun_ran_gaussian_pdf(data[n], Gauss2);
     double prob_xn = p1_n + p2_n;
 
     double p_1_given_xn = p1_n / prob_xn;
     double p_2_given_xn = p2_n / prob_xn;
 
     dL_d_mixCof += (p1_n - p2_n) / prob_xn;
     if (sigma1 < 1e-10) sigma1 = 1e-10;
     if (sigma2 < 1e-10) sigma2 = 1e-10;
     dL_d_mu1 += p_1_given_xn * (data[n] - mu1) / (sigma1 * sigma1);
     dL_d_mu2 += p_2_given_xn * (data[n] - mu2) / (sigma2 * sigma2);
   }
 
   params[0] += 0.1 * dL_d_mixCof;
   params[1] += 0.1 * dL_d_mu1;
   params[3] += 0.1 * dL_d_mu2;
 
   if (params[0] < 0.01) params[0] = 0.01;
   if (params[0] > 0.99) params[0] = 0.99;
 
   // 限制均值和標準差的範圍 (範例)
   if (params[1] < data_sample_mean() - 10) params[1] = data_sample_mean() - 10;
   if (params[1] > data_sample_mean() + 10) params[1] = data_sample_mean() + 10;
   if (params[3] < data_sample_mean() - 10) params[3] = data_sample_mean() - 10;
   if (params[3] > data_sample_mean() + 10) params[3] = data_sample_mean() + 10;
   if (params[2] < 0.1) params[2] = 0.1;
   if (params[2] > 10) params[2] = 10;
   if (params[4] < 0.1) params[4] = 0.1;
   if (params[4] > 10) params[4] = 10;
   
}

// Function to estimate the maximum likelihood for the 2-component GMM
double data_Gauss2_maxLikelihood() {
  // Initial parameter guesses (you might want to initialize these more intelligently)
  double params[5] = {0.5, data_sample_mean() - 1.0, 1.0, data_sample_mean() + 1.0,
                     1.0}; // mixCof, mu1, sigma1, mu2, sigma2

  const int max_iterations = 100;
  const double tolerance = 1e-5;
  double prev_logL = -INFINITY;
  double logL = 0.0;
  double likelihood = 0.0;

  for (int iter = 0; iter < max_iterations; ++iter) {
   update_params(params, data, dataN);

   double variance = data_sample_variance();
   if (variance < 0) variance = 0;
   params[2] = sqrt(variance) / 2;
   params[4] = sqrt(variance) / 2;

   logL = log_likelihood_2Gauss(params, data, dataN);

   if (fabs(logL - prev_logL) < tolerance) {
     break;
   }
   prev_logL = logL;
 }

 likelihood = exp(logL);
 return likelihood;
 //return logL;
}

/* Compute Riemann sum to approximate the integral
 *
 * ∫ μ,σ  P[D,μ,σ]
 *
 */
double data_prob_1component_bySumming() {
  double prob_total = 0.0;
  for (uint m = 0; m < cdf_Gauss_n; ++m) {
    double mu = cdfInv_Gauss[m];
    for (uint s = 0; s < cdf_gamma_n; ++s) {
      double sigma = sigma_of_precision(cdfInv_gamma[s]);
      Gauss_params cur_params = {mu, sigma};
      prob_total += prob_data_given_1Gauss(cur_params);
    }
  }
  return prob_total / (double)(cdf_Gauss_n * cdf_gamma_n);
}

/* Compute Riemann sum to approximate integral
 *
 * ∫ m,μ₁,σ₁,μ₂,σ₂  P[D,m,μ₁,σ₁,μ₂,σ₂]
 *
 */
double data_prob_2component_bySumming() {
  double prob_total = 0.0;

  for (uint m1 = 0; m1 < cdf_Gauss_n; ++m1) {
    double mu1 = cdfInv_Gauss[m1];
    for (uint m2 = 0; m2 < cdf_Gauss_n; ++m2) {
      double mu2 = cdfInv_Gauss[m2];
      for (uint s1 = 0; s1 < cdf_gamma_n; ++s1) {
        double sigma1 = sigma_of_precision(cdfInv_gamma[s1]);
        Gauss_params cur_params1 = {mu1, sigma1};
        for (uint s2 = 0; s2 < cdf_gamma_n; ++s2) {
          double sigma2 = sigma_of_precision(cdfInv_gamma[s2]);
          Gauss_params cur_params2 = {mu2, sigma2};
          for (uint mi = 0; mi < cdf_JBeta_n; ++mi) {
            double mixCof = cdfInv_JBeta[mi];
            prob_total += prob_data_given_2Gauss(mixCof, cur_params1, cur_params2);
          }
        }
      }
    }
  }
  return prob_total /
         (double)(cdf_Gauss_n * cdf_Gauss_n * cdf_gamma_n * cdf_gamma_n *
                  cdf_JBeta_n);
}

/* Use sampling to estimate
 * ∫ μ,σ  P[D,μ,σ]
 */
double data_prob_1component_bySampling() { // 抽樣法
  double prob_total = 0.0;

  for (uint iter = 0; iter < sampleRepeatNum; ++iter) {
    Gauss_params params = prior_Gauss_params_sample(); // 產生一組可能的高斯分布參數
    prob_total += prob_data_given_1Gauss(params);     // 計算在這個參數下的數據概率
  }
  return prob_total / (double)sampleRepeatNum;       // 平均值近似積分結果
}

/* Use sampling to estimate
 * ∫ m,μ₁,σ₁,μ₂,σ₂  P[D,m,μ₁,σ₁,μ₂,σ₂]
 */
double data_prob_2component_bySampling() {
  double prob_total = 0.0;

  for (uint iter = 0; iter < sampleRepeatNum; ++iter) {
    Gauss_mixture_params params = prior_Gauss_mixture_params_sample();
    prob_total += prob_data_given_2Gauss(params.mixCof, params.Gauss1,
                                         params.Gauss2);
  }
  return prob_total / (double)sampleRepeatNum;
}

int main(int argc, char *argv[]) {
   uint datasets_n = 10;
 
   {
     char usage_fmt[] = "Usage: %s [num_datasets]\n";
     switch (argc) {
     case 1:
       break;
     case 2:
       datasets_n = atoi(argv[1]);
       if (!datasets_n) {
         printf(usage_fmt, argv[0]);
         exit(64);
       }
       break;
     default:
       printf(usage_fmt, argv[0]);
       exit(64);
     }
   }
 
   GSLfun_setup();
   double prob_data1_bySampling, prob_data2_bySampling;
   double prob_data1_bySumming, prob_data2_bySumming;
   double maxL_1Gauss, maxL_2Gauss;
 
   cdfInv_precompute();
 
   uint model1_sampling_favors1 = 0;
   uint model1_summing__favors1 = 0;
   uint model2_sampling_favors1 = 0;
   uint model2_summing__favors1 = 0;
 
   printf("Starting computation for %d datasets each. ...\n", datasets_n);
 
   printf("\nData generated with one component\n");
   for (uint iter = 0; iter < datasets_n; ++iter) {
     Gauss_params model_params = prior_Gauss_params_sample();
     printf("generating data with: (μ,σ) =  (%4.2f,%4.2f)\n", model_params.mu,
            model_params.sigma);
     data_generate_1component(model_params); // 產生一組可能的高斯分布參數
     maxL_1Gauss = data_Gauss1_maxLikelihood();
     maxL_2Gauss = data_Gauss2_maxLikelihood();
     printf("Data maximum likelihood:\n");
     printf("  One component model= %g\n", maxL_1Gauss);
     printf("  Two component model= %g\n", maxL_2Gauss);
     if (maxL_1Gauss > maxL_2Gauss)
       printf("  One component model favored by Maximum Likelihood\n");
     else
       printf("  Two component model favored by Maximum Likelihood\n");
 
     //邊際機率evidence
     prob_data1_bySampling = data_prob_1component_bySampling();
     prob_data2_bySampling = data_prob_2component_bySampling();
     prob_data1_bySumming = data_prob_1component_bySumming();
     prob_data2_bySumming = data_prob_2component_bySumming();
     printf("Integrals by sampling= (%g,%g)  by summing: (%g,%g)\n\n",
            prob_data1_bySampling, prob_data2_bySampling, prob_data1_bySumming,
            prob_data2_bySumming);
     if (prob_data1_bySampling > prob_data2_bySampling)
       ++model1_sampling_favors1; // 單一高斯模型的邊際機率 > 兩成分混合模型的邊際機率時，模型1勝出，累計勝利次數
     if (prob_data1_bySumming > prob_data2_bySumming)
       ++model1_summing__favors1;
   }
 
   printf("\nData generated with two components\n");
   for (uint iter = 0; iter < datasets_n; ++iter) {
     Gauss_mixture_params model_params = prior_Gauss_mixture_params_sample();
     printf(
         "generating data with:  m; (μ1,σ1); (μ2,σ2) =  %5.3f; (%4.2f,%4.2f); "
         "(%4.2f,%4.2f)\n",
         model_params.mixCof, model_params.Gauss1.mu, model_params.Gauss1.sigma,
         model_params.Gauss2.mu, model_params.Gauss2.sigma);
     data_generate_2component(model_params); // 決定要用哪個分布
     maxL_1Gauss = data_Gauss1_maxLikelihood();
     maxL_2Gauss = data_Gauss2_maxLikelihood();
     printf("Data maximum likelihood:\n");
     printf("  One component model= %g\n", maxL_1Gauss);
     printf("  Two component model= %g\n", maxL_2Gauss);
     if (maxL_1Gauss > maxL_2Gauss)
       printf("  One component model favored by Maximum Likelihood\n");
     else
       printf("  Two component model favored by Maximum Likelihood\n");
 
     prob_data1_bySampling = data_prob_1component_bySampling();
     prob_data2_bySampling = data_prob_2component_bySampling();
     prob_data1_bySumming = data_prob_1component_bySumming();
     prob_data2_bySumming = data_prob_2component_bySumming();
     printf("Integrals by sampling= (%g,%g)  by summing: (%g,%g)\n\n",
            prob_data1_bySampling, prob_data2_bySampling, prob_data1_bySumming,
            prob_data2_bySumming);
     if (prob_data1_bySampling > prob_data2_bySampling)
       ++model2_sampling_favors1;
     if (prob_data1_bySumming > prob_data2_bySumming)
       ++model2_summing__favors1;
   }
 
   printf("By sampling: Model1 data, correct selection %u/%u\n",
          model1_sampling_favors1, datasets_n);
   printf("             Model2 data, correct selection %u/%u\n",
          (datasets_n - model2_sampling_favors1), datasets_n);
   printf("By summing:  Model1 data, correct selection %u/%u\n",
          model1_summing__favors1, datasets_n);
   printf("             Model2 data, correct selection %u/%u\n",
          (datasets_n - model2_summing__favors1), datasets_n);
 
   return 0;
 }