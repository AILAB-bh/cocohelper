{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
   :show-inheritance:


   {% block methods %}
   {% if methods %}

   ..
      REMOVE THIS COMMENTO TO ADD INIT, REMEMBER TO CHANGE autoclass_content IN conf.py
      .. automethod:: __init__
         :noindex:


   .. rubric:: {{ _('Method List') }}
   .. autosummary::
      :recursive:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}



   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes List') }}
   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}



   {% block methods_details %}
   .. rubric:: {{ _('Methods Details') }}
   {% for item in methods %}
   .. automethod::
      {{ item }}
   {%- endfor %}
   {% endblock %}


   {% block attributes_details %}
   {% if attributes %}
   .. rubric:: {{ _('Attribute Details') }}

   {% for item in attributes %}
   .. autoattribute::
       {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

