import React from 'react'

function Header() {
  return (
    <div className='flex justify-between lg:p-5 p-3 container mx-auto text-morado'>
      <div className='flex lg:flex-row flex-col lg:justify-between w-full items-center'>
        <span className='font-secondary font-extrabold block lg:text-5xl text-4xl'>Proyecto<span className='text-naranja'>Tesis</span></span>
        
        <div className='flex lg:flex-row flex-col justify-center items-center text-xl font-bold lg:gap-4'>
          <a className='p-4 cursor-pointer'>Home</a>
          <a className='py-2 px-8 border-black border-2 cursor-pointer'>Noticias</a>
          <a className='p-4' href='https://wa.link/1ayc0i' target='_blank'>
            <button type='button' className='font-tertiary bg-celeste py-2 px-8 cursor-pointer'>Predecir!</button>
          </a>
        </div>
      </div>
    </div>
  )
}

export default Header