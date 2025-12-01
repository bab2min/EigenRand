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

#if EIGEN_VERSION_AT_LEAST(5,0,0)
#define EIGENRAND_EIGEN_50_MODE
#define EIGENRAND_EIGEN_34_MODE
#elif EIGEN_VERSION_AT_LEAST(3,3,10)
#define EIGENRAND_EIGEN_34_MODE
#elif EIGEN_VERSION_AT_LEAST(3,3,4)
#define EIGENRAND_EIGEN_33_MODE
#endif

#if EIGEN_VERSION_AT_LEAST(3,3,4)
// Eigen 3.3.4 or later is required.
#else
#error Eigen 3.3.4 or later is required.
#endif

#endif
