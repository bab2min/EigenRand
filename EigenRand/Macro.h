/**
 * @file Macro.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.2.1
 * @date 2020-07-11
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef EIGENRAND_MACRO_H
#define EIGENRAND_MACRO_H

#define EIGENRAND_WORLD_VERSION 0
#define EIGENRAND_MAJOR_VERSION 2
#define EIGENRAND_MINOR_VERSION 0

#if EIGEN_VERSION_AT_LEAST(3,3,7)
#else
#error At least Eigen 3.3.7 is required.
#endif

#endif