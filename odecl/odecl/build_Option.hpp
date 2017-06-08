//---------------------------------------------------------------------------//
// Copyright (c) 2015,2016 Eleftherios Avramidis <el.avramidis@gmail.com>
//
// Distributed under The MIT License (MIT)
// See accompanying file LICENSE.txt
//---------------------------------------------------------------------------//

#ifndef odecl_BUILD_OPTION_HPP
#define odecl_BUILD_OPTION_HPP

namespace odecl
{
	enum class build_Option
	{
		FastRelaxedMath = 1,
		stdCL20 = 2,
		stdCL21 = 3
	};
}


#endif // odecl_BUILD_OPTION_HPP
