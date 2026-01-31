/**
 * @file Macro.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.6.0
 * @date 2026-01-31
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_MACRO_H
#define EIGENRAND_MACRO_H

#define EIGENRAND_WORLD_VERSION 0
#define EIGENRAND_MAJOR_VERSION 6
#define EIGENRAND_MINOR_VERSION 0

 // Eigen version detection
 // - Eigen 5.x uses Semantic Versioning (MAJOR.MINOR.PATCH)
 // - Eigen 3.5+ (development branch) also uses eigen_packet_wrapper like Eigen 5.x
 // - Eigen 5.0 is the successor to Eigen 3.4

#if EIGEN_VERSION_AT_LEAST(5,0,0)
// Eigen 5.x (SemVer)
#define EIGENRAND_EIGEN_34_MODE
#define EIGENRAND_EIGEN_50_MODE
#elif EIGEN_VERSION_AT_LEAST(3,5,0)
// Eigen 3.5+ (development branch before 5.0, uses eigen_packet_wrapper)
#define EIGENRAND_EIGEN_34_MODE
#define EIGENRAND_EIGEN_50_MODE
#elif EIGEN_VERSION_AT_LEAST(3,3,10)
// Eigen 3.3.10 ~ 3.4.x
#define EIGENRAND_EIGEN_34_MODE
#elif EIGEN_VERSION_AT_LEAST(3,3,4)
// Eigen 3.3.4 ~ 3.3.9
#define EIGENRAND_EIGEN_33_MODE
#endif

// Support Eigen 3.3.4 ~ 3.4.x, Eigen 3.5+ (dev), and Eigen 5.x
#if EIGEN_VERSION_AT_LEAST(5,0,0)
// Eigen 5.x: OK
#elif EIGEN_VERSION_AT_LEAST(3,3,4)
// Eigen 3.3.4 ~ 3.5.x: OK
#else
#error Eigen 3.3.4 or higher is required.
#endif

#endif
