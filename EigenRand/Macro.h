/**
 * @file Macro.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.5.1
 * @date 2024-09-08
 * 
 * @copyright Copyright (c) 2020-2021
 * 
 */

#ifndef EIGENRAND_MACRO_H
#define EIGENRAND_MACRO_H

#define EIGENRAND_WORLD_VERSION 0
#define EIGENRAND_MAJOR_VERSION 5
#define EIGENRAND_MINOR_VERSION 1

// Eigen 5.x uses Semantic Versioning (MAJOR.MINOR.PATCH)
// Eigen 5.0 is the successor to Eigen 3.4, with similar internal APIs
#if EIGEN_VERSION_AT_LEAST(5,0,0)
#define EIGENRAND_EIGEN_34_MODE
#define EIGENRAND_EIGEN_50_MODE
#elif EIGEN_VERSION_AT_LEAST(3,3,10)
#define EIGENRAND_EIGEN_34_MODE
#elif EIGEN_VERSION_AT_LEAST(3,3,4)
#define EIGENRAND_EIGEN_33_MODE
#endif

// Support Eigen 3.3.4 ~ 3.4.x and Eigen 5.x
// Note: Eigen skipped version 4.x and went from 3.4 to 5.0
#if EIGEN_VERSION_AT_LEAST(5,0,0)
// Eigen 5.x: OK (successor to 3.4)
#elif EIGEN_VERSION_AT_LEAST(3,3,4)
// Eigen 3.3.4 ~ 3.4.x: OK
#else
#error Eigen 3.3.4 ~ 3.4.x or Eigen 5.x is required.
#endif

#endif
