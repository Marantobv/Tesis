import React from 'react'
import { motion } from 'framer-motion'
import { fade } from '../utils/variants'
import CountUp from 'react-countup';
import { useInView } from 'react-intersection-observer'
import stock3 from './../img/stock3.avif'
import results from './../img/real_vs_predicted_prices.png'

function About() {

  const {ref,inView} = useInView({

  })

  return (
    <div className='section box-border' id='about'>
      <div className='container lg:flex mx-auto gap-4'>
        {/* <motion.div variants={fade("right", 1)} initial={"init"} whileInView={"view"} className='flex-1 mx-auto h-[200px] w-[200px] lg:h-[600px]'>
          <h1>IMAGEN</h1>
        </motion.div> */}
        <div className='m-auto flex-[2]'>
          <motion.img variants={fade("down", 1)} initial={"init"} whileInView={"view"} className='hidden lg:flex' src={results} alt='Stock Market'></motion.img>
        </div>
        <motion.div variants={fade("left", 1)} initial={"init"} whileInView={"view"} className='flex flex-col justify-center text-left p-5 flex-[1]'>
          <h2 className='text-5xl font-primary font-bold text-amarillo uppercase'>Proyecto 20241094</h2>
          <h3 className='text-morado text-2xl font-secondary font-bold mt-4'>Lorem Ipsum is simply dummy text of the printing and typesetting industry.</h3>
          <p className='text-morado font-secondary text-lg mt-3 font-light break-words'>Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s</p>
          <div className='flex gap-10 mt-6'>
            <div ref={ref}>
            <span className='font-primary text-verde text-4xl font-bold'>0.</span>{inView ? <CountUp className='font-primary text-verde text-4xl font-bold' end={79}></CountUp> : null}
              <p className='text-morado text-xl font-tertiary font-light'>MAPE</p>
            </div>
            <div>
            {inView ? <CountUp className='font-primary text-verde text-4xl font-bold' end={42}></CountUp> : null}<span className='font-primary text-verde text-4xl font-bold'>.14</span>
              <p className='text-morado text-xl font-tertiary font-light'>MAE</p>
            </div>
            <div>
            {inView ? <CountUp className='font-primary text-verde text-4xl font-bold' end={51}></CountUp> : null}<span className='font-primary text-verde text-4xl font-bold'>.92</span>
              <p className='text-morado text-xl font-tertiary font-light'>RMSE</p>
            </div>
          </div>
            <div className='mt-10 flex items-center gap-4'>
              <a href='#' target='_blank'>
                <button className='btn font-tertiary gradient' type='button'>Predecir</button>
              </a>
              
              <a target='_blank' href='https://drive.google.com/drive/folders/1P5F5IFjEl2L8CnrFF1rivHibRf03nmVC?usp=sharing' className='text-verde hover:tracking-wider transition-all cursor-pointer text-xl font-bold'>Datos usados</a>

            </div>

        </motion.div>
      </div>
    </div>
  )
}

export default About