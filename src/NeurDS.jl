module NeurDS

using ModelingToolkit, OrdinaryDiffEq, Plots

# helper functions for ODE system
r(v)= 127v/105 +8265/105
m∞(v) = 1/(exp((-22-v)/8)+1)
H(x) = x>0. ? 1. : 0.

function PlantCell(;name)
    @parameters μh μn μx μCa gna gn gx gCa xs Cas gleak
    sts = @variables t v(t) n(t) h(t) x(t) Ca(t) Iext(t)
    D = Differential(t)

    ODESystem(Equation[
        D(v) ~ gna*^(m∞(v),3.0)*h*(30-v)+gn*^(n,4.0)*(-75 -v)+gx*x*(30-v)+gCa*Ca/(.5+Ca)*(-75-v)+gleak*(-40-v) + Iext,
        D(h) ~ μh*((1-h)*(.07*exp((25-r(v))/20))-h*(1/(1+exp((55-r(v))/10)))),
        D(n) ~ μn*((1-n)*(.01*(55-r(v)))/(exp((55-r(v))/10)-1)-n*0.125*exp((45 - r(v))/80)),
        D(x) ~ μx*((1/(exp(.15*(-50 + xs -v))+1))-x),
        D(Ca) ~ μCa*(.0085*x*(140 - v +Cas)-Ca),
    ]; name, defaults = Dict([
        #Initial Conditions
        v => 0.,
        n => 0.,
        h => 0.,
        x => 0.9,
        Ca => .7,
        #parameters
        μh => .08,
        μn => .08,
        gna => 4.,
        gn => 3.,
        xs => -4.,
        Cas => -60.,
        gleak => .0025,
        μx => .012,
        μCa => .00025
    ]))
end

@named c1 = PlantCell()
@named c2 = PlantCell()

struct AlphaSynapse
    sys
    pre
    post
end

function AlphaSynapse(pre, post; name)
    @parameters α β threshold k E g
    sts = @variables t s(t) v(t) vpre(t)
    D = Differential(t)
    sys = ODESystem(Equation[
        D(s) ~ α*(1-s)/(1+exp(threshold-vpre)/k) - β*s,
        v ~ g*s*(E-v)
    ]; name, defaults = Dict([
        #initial conditions
        s => 0.,
        #parameters
        α => .1,
        β => .1,
        threshold => 0.,
        k => 0.,
        E => 30.,
        g => .1
    ]))
    AlphaSynapse(sys, pre, post)
end

@named s = AlphaSynapse(c1, c2)

function network(cells, synapses; name)
    external_currents = []
    for cell in cells
        iexts = [syn.sys.v for syn in synapses if syn.post.name == cell.name]
        push!(external_currents, cell.Iext ~ isempty(iexts) ? 0 : sum(iexts))
    end
    presynaptic_drives = [syn.sys.vpre ~ syn.pre.v for syn in synapses]
    connections = vcat(external_currents, presynaptic_drives)

    compose(ODESystem(connections), cells..., [syn.sys for syn in synapses]...)
end

@named net = network([c1,c2], [s])

structural_simplify(net)
