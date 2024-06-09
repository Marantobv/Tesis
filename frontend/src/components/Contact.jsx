import React from 'react'
import { motion } from 'framer-motion'
import { fade } from '../utils/variants'

function Contact() {
  return (
    <div className='section' id='contact'>
      <div className='container mx-auto max-w-[1200px]'>
        <div className='flex lg:flex-row flex-col justify-center items-center gap-[40px]'>
          <motion.div variants={fade("right", 1)} initial={"init"} whileInView={"view"} className='flex-1'>
            <p className='font-secondary font-bold text-[#C637A0] lg:text-4xl text-3xl py-3'>Ponte en contacto</p>
            <h2 className='font-secondary text-6xl lg:text-8xl text-white'>Trabajemos juntos!</h2>
          </motion.div>
          <motion.div variants={fade("left", 1)} initial={"init"} whileInView={"view"} className='border border-[#C637A0] flex flex-col rounded-[20px] p-[10px] flex-1 w-full'>
            <div className='p-3'>
              <input type='text' placeholder='Tu nombre' className='input-form'></input>
            </div>
            <div className='p-3'>
              <input placeholder='Tu correo' className='input-form'></input>
            </div>
            <div className='p-3 mt-10'>
              <textarea placeholder='Mensaje' className='input-form resize-none h-[100px]'></textarea>
            </div>
            <div className='mt-[50px] p-3'>
              <button className='btn font-tertiary gradient'>Contáctame</button>
            </div>
          </motion.div>
        </div>
        <h2 className='text-center font-secondary text-3xl mt-[50px] text-white uppercase'>(En progreso...)</h2>
      </div>
    </div>
  )
}

export default Contact