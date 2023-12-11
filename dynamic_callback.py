def update_beam_energy_options(selected_elements):
    # Check which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    beam_energies = set()

    print(selected_elements)

    for key in testf['run_01062020']:
        ds = testf['run_01062020'][key]
        element = ds.attrs.get('Element')
        beam_energy = ds.attrs.get('Beam E eV')

        if isinstance(element, str) and element.strip() in selected_elements:
            if isinstance(beam_energy, str):
                beam_energies.add(beam_energy.strip())
            elif isinstance(beam_energy, float):
                beam_energies.add(str(beam_energy).strip())

    # Convert strings in beam_energies to float using a list comprehension
    beam_energies = [float(energy) for energy in beam_energies]

    # Convert the set to a sorted list to maintain a consistent order
    beam_energy_options = [{'label': energy, 'value': energy} for energy in sorted(beam_energies)]

    return beam_energy_options
