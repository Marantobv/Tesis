import React from 'react'
import { motion } from 'framer-motion'
import { fade } from '../utils/variants'
import CountUp from 'react-countup';
import { useInView } from 'react-intersection-observer'
import stock3 from './../img/stock3.avif'

function About() {

  const {ref,inView} = useInView({

  })

  return (
    <div className='section box-border' id='about'>
      <div className='lg:flex mx-auto gap-[150px] max-w-[1000px]'>
        {/* <motion.div variants={fade("right", 1)} initial={"init"} whileInView={"view"} className='flex-1 mx-auto h-[200px] w-[200px] lg:h-[600px]'>
          <h1>IMAGEN</h1>
        </motion.div> */}
        <div className='m-auto'>
          <motion.img variants={fade("down", 1)} initial={"init"} whileInView={"view"} className='hidden lg:flex' src={stock3} alt='Stock Market'></motion.img>
        </div>
        <motion.div variants={fade("left", 1)} initial={"init"} whileInView={"view"} className='flex flex-col justify-center text-left p-5 flex-[2]'>
          <h2 className='text-4xl font-primary font-bold text-amarillo uppercase'>ProyectoTesis</h2>
          <h3 className='text-morado text-2xl font-secondary font-bold mt-4'>Lorem Ipsum is simply dummy text of the printing and typesetting industry.</h3>
          <p className='text-morado font-secondary text-lg mt-3 font-light break-words'>Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s</p>
          <div className='flex gap-10 mt-6'>
            <div ref={ref}>
              {inView ? <CountUp className='font-secondary gradient-text text-5xl font-bold' end={8}></CountUp> : null}<span className='font-secondary gradient-text text-xl font-bold'>vo</span>
              <p className='text-morado text-xl font-tertiary font-light'>Ciclo</p>
            </div>
            <div>
              {inView ? <CountUp className='font-secondary gradient-text text-5xl font-bold' end={10}></CountUp> : null}
              <p className='text-morado text-xl font-tertiary font-light'>Proyectos</p>
            </div>
          </div>
            <div className='mt-10 flex items-center gap-4'>
              <a href='https://wa.link/1ayc0i' target='_blank'>
                <button className='btn font-tertiary gradient' type='button'>Predecir</button>
              </a>
              
              <a target='_blank' href='/CONSTANCIA_71230817.pdf' className='gradient-text hover:tracking-wider transition-all cursor-pointer text-xl font-bold'>Datos</a>

            </div>

        </motion.div>
      </div>
    </div>
  )
}

export default About