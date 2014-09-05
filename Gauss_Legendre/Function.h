//-----------------------------------------------------------------------
// <copyright file="Function.h" company="dc1394's software">
//     Copyright ©  2014 @dc1394 All Rights Reserved.
// </copyright>
//-----------------------------------------------------------------------
#pragma once

namespace myfunctional {
    //! A template class.
    /*!
    std::function<double (double)>の代わりになるtemplate class
    */
    template <typename FUNCTYPE>
    class Function final
    {
        //! A private const variable.
        /*!
        operator()で呼び出す関数
        */
        const FUNCTYPE & func_;

    public:
        //! A constructor.
        /*!
        operator()で呼び出す関数
        */
        Function(const FUNCTYPE & func) : func_(func) {}
        
        //! A public member function.
        /*!
        operator()の宣言
        関数f(x)の値を返す
        \param x xの値
        \return f(x)の値
        */
        double operator()(double x) const
        {
            return func_(x);
        }
    };

    //! A template function（非メンバ関数）.
    /*!
    Function<FUNCTYPE>を生成する
    \param func 格納する関数
    \return 生成されたFunction<FUNCTYPE>
    */
    template <class FUNCTYPE>
    Function<FUNCTYPE> make_function(const FUNCTYPE & func)
    {
        return Function<FUNCTYPE>(func);
    }
}
